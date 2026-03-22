#include "ZeroCopyIngest.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "Contracts.hpp"
#include "EventSynthesizer.hpp"

#ifndef VISION_ENGINE_USE_ZMQ
#define VISION_ENGINE_USE_ZMQ 1
#endif

#if VISION_ENGINE_USE_ZMQ
#include <zmq.h>
#endif

namespace vision {

namespace {

constexpr const char* kDecisionEndpoint = "tcp://127.0.0.1:5555";
constexpr const char* kVideoIngressEndpoint = "tcp://127.0.0.1:6000";
constexpr uint64_t kFallbackIngressStride = 4;

} // namespace

ZeroCopyIngest::ZeroCopyIngest(uint32_t width_px, uint32_t height_px, uint32_t channels)
    : zmq_context_(nullptr),
      zmq_publisher_(nullptr),
      zmq_subscriber_(nullptr),
      w_(width_px),
      h_(height_px),
      c_(channels) {
#if VISION_ENGINE_USE_ZMQ
    zmq_context_ = zmq_ctx_new();
    if (!zmq_context_) {
        std::cerr << "[Vision::NVMM] Failed to initialize ZeroMQ context. Falling back to deterministic local ingest.\n";
        return;
    }

    zmq_publisher_ = zmq_socket(zmq_context_, ZMQ_PUB);
    zmq_subscriber_ = zmq_socket(zmq_context_, ZMQ_SUB);

    if (!zmq_publisher_ || !zmq_subscriber_) {
        std::cerr << "[Vision::NVMM] Failed to allocate ZeroMQ sockets. Local fallback ingest remains available.\n";
        return;
    }

    int linger_ms = 0;
    int no_wait_timeout = 0;
    int send_hwm = 1;
    zmq_setsockopt(zmq_publisher_, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));
    zmq_setsockopt(zmq_publisher_, ZMQ_SNDHWM, &send_hwm, sizeof(send_hwm));

    zmq_setsockopt(zmq_subscriber_, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));
    zmq_setsockopt(zmq_subscriber_, ZMQ_RCVTIMEO, &no_wait_timeout, sizeof(no_wait_timeout));
    zmq_setsockopt(zmq_subscriber_, ZMQ_SUBSCRIBE, "", 0);

    if (zmq_bind(zmq_publisher_, kDecisionEndpoint) != 0) {
        std::cerr << "[Vision::NVMM] Failed to bind ZeroMQ publisher on " << kDecisionEndpoint
                  << ". Event dispatch will stay local.\n";
        zmq_close(zmq_publisher_);
        zmq_publisher_ = nullptr;
    }

    if (zmq_connect(zmq_subscriber_, kVideoIngressEndpoint) != 0) {
        std::cerr << "[Vision::NVMM] Failed to connect upstream stream orchestrator on "
                  << kVideoIngressEndpoint << ". Using deterministic local fallback ingress.\n";
        zmq_close(zmq_subscriber_);
        zmq_subscriber_ = nullptr;
    }
#endif
}

ZeroCopyIngest::~ZeroCopyIngest() {
#if VISION_ENGINE_USE_ZMQ
    if (zmq_subscriber_) {
        zmq_close(zmq_subscriber_);
    }
    if (zmq_publisher_) {
        zmq_close(zmq_publisher_);
    }
    if (zmq_context_) {
        zmq_ctx_destroy(zmq_context_);
    }
#endif
    releaseMemory();
}

bool ZeroCopyIngest::allocateMemory() {
    last_frame_id_ = 0;
    fallback_frame_cursor_ = 0;
    last_event_tick_ = 0;
    last_event_score_ = 0.0f;
    using_fallback_ingest_ = true;

    // In actual production:
    // cudaMallocHost(&p_host_pinned, size_bytes);
    // cudaHostGetDevicePointer(&d_frame_current_, p_host_pinned, 0);
    return true;
}

void ZeroCopyIngest::releaseMemory() {
    // cudaFreeHost(p_host_pinned);
}

bool ZeroCopyIngest::pollNetworkInterface() {
#if VISION_ENGINE_USE_ZMQ
    if (zmq_subscriber_) {
        UpstreamFrameEnvelope envelope{};
        const int received = zmq_recv(zmq_subscriber_, &envelope, sizeof(envelope), ZMQ_DONTWAIT);
        if (received == sizeof(envelope)) {
            if (!is_expected_frame_shape(envelope, w_, h_, c_)) {
                std::cerr << "[Vision::NVMM] Dropped upstream frame envelope with unexpected geometry: "
                          << envelope.width << "x" << envelope.height << "x" << envelope.channels << "\n";
                return false;
            }

            last_frame_id_ = envelope.frame_id;
            using_fallback_ingest_ = false;
            return true;
        }
    }
#endif

    // Deterministic fallback ingress keeps the engine runnable even when #3 is offline.
    ++fallback_frame_cursor_;
    if (fallback_frame_cursor_ % kFallbackIngressStride == 0) {
        last_frame_id_ = fallback_frame_cursor_;
        using_fallback_ingest_ = true;
        return true;
    }

    return false;
}

void ZeroCopyIngest::processTemporalDerivative(uint64_t clock_tick) {
    last_event_tick_ = clock_tick;
    last_event_score_ = compute_temporal_score(clock_tick, last_frame_id_, using_fallback_ingest_);
    event_triggered_.store(last_event_score_ >= detection_threshold_, std::memory_order_release);
}

bool ZeroCopyIngest::hasThresholdEvent() const {
    return event_triggered_.load(std::memory_order_acquire);
}

void ZeroCopyIngest::dispatchInferenceEvent(uint64_t clock_tick) {
    const auto snapshot = synthesize_event(
        clock_tick,
        last_frame_id_,
        using_fallback_ingest_,
        next_object_id_.fetch_add(1, std::memory_order_relaxed)
    );
    const auto& payload = snapshot.payload;

    std::cout << "[Vision::NVMM] Event tick=" << last_event_tick_
              << " frame=" << last_frame_id_
              << " object=" << payload.object_id
              << " confidence=" << payload.confidence
              << (using_fallback_ingest_ ? " [fallback_ingress]" : " [upstream_ingress]")
              << "\n";

#if VISION_ENGINE_USE_ZMQ
    if (zmq_publisher_) {
        zmq_msg_t msg;
        zmq_msg_init_size(&msg, sizeof(InferencePayload));
        std::memcpy(zmq_msg_data(&msg), &payload, sizeof(InferencePayload));

        const int send_result = zmq_msg_send(&msg, zmq_publisher_, ZMQ_DONTWAIT);
        zmq_msg_close(&msg);

        if (send_result == sizeof(InferencePayload)) {
            std::cout << "[Vision::NVMM] Dispatched event to Decision Bus via ZeroMQ.\n";
        } else {
            std::cerr << "[Vision::NVMM] ZeroMQ dispatch skipped or back-pressured. Event remained local.\n";
        }
        return;
    }
#endif

    std::cout << "[Vision::NVMM] ZeroMQ disabled or unavailable. Event retained in local debug stream.\n";
}

float ZeroCopyIngest::lastEventScore() const {
    return last_event_score_;
}

bool ZeroCopyIngest::usingFallbackIngress() const {
    return using_fallback_ingest_;
}

uint64_t ZeroCopyIngest::lastFrameId() const {
    return last_frame_id_;
}

} // namespace vision
