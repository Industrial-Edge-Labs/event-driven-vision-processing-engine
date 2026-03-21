#include "ZeroCopyIngest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

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

#pragma pack(push, 1)
struct InferencePayload {
    uint64_t timestamp;
    uint32_t object_id;
    float confidence;
    float x;
    float y;
    float dx;
    float dy;
};

struct UpstreamFrameEnvelope {
    uint64_t timestamp;
    uint64_t frame_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
};
#pragma pack(pop)

static_assert(sizeof(InferencePayload) == 32, "InferencePayload wire format changed unexpectedly.");

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count());
}

float clamp_unit(float value) {
    return std::clamp(value, 0.0f, 0.999f);
}

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
        std::cerr << "[Vision::NVMM] Failed to initialize ZeroMQ context. Falling back to synthetic ingest only.\n";
        return;
    }

    zmq_publisher_ = zmq_socket(zmq_context_, ZMQ_PUB);
    zmq_subscriber_ = zmq_socket(zmq_context_, ZMQ_SUB);

    if (!zmq_publisher_ || !zmq_subscriber_) {
        std::cerr << "[Vision::NVMM] Failed to allocate ZeroMQ sockets. Synthetic ingest remains available.\n";
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
                  << kVideoIngressEndpoint << ". Using synthetic frame ingress.\n";
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
    synthetic_frame_cursor_ = 0;
    last_event_tick_ = 0;
    last_event_score_ = 0.0f;
    using_synthetic_ingest_ = true;

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
            last_frame_id_ = envelope.frame_id;
            using_synthetic_ingest_ = false;
            return true;
        }
    }
#endif

    // Synthetic ingress keeps the engine debuggable even without #3 online.
    ++synthetic_frame_cursor_;
    if (synthetic_frame_cursor_ % 4 == 0) {
        last_frame_id_ = synthetic_frame_cursor_;
        using_synthetic_ingest_ = true;
        return true;
    }

    return false;
}

void ZeroCopyIngest::processTemporalDerivative(uint64_t clock_tick) {
    const double phase = static_cast<double>((clock_tick + last_frame_id_) % 720) * 0.021;
    const float temporal_energy = static_cast<float>(0.45 + 0.35 * std::sin(phase));
    const float transport_bias = using_synthetic_ingest_ ? 0.02f : 0.08f;
    const float edge_density = static_cast<float>(0.06 + 0.06 * std::cos(phase * 0.5));

    last_event_tick_ = clock_tick;
    last_event_score_ = clamp_unit(temporal_energy + transport_bias + edge_density);
    event_triggered_.store(last_event_score_ >= detection_threshold_, std::memory_order_release);
}

bool ZeroCopyIngest::hasThresholdEvent() const {
    return event_triggered_.load(std::memory_order_acquire);
}

void ZeroCopyIngest::dispatchInferenceEvent(uint64_t clock_tick) {
    InferencePayload payload{};
    const double phase = static_cast<double>((clock_tick + last_frame_id_) % 360) * 0.017453292519943295;

    payload.timestamp = now_ns();
    payload.object_id = next_object_id_.fetch_add(1, std::memory_order_relaxed);
    payload.confidence = clamp_unit(0.55f + (last_event_score_ * 0.35f));
    payload.x = 960.0f + static_cast<float>(420.0 * std::sin(phase));
    payload.y = 540.0f + static_cast<float>(180.0 * std::cos(phase * 0.5));
    payload.dx = 4.0f + static_cast<float>(18.0 * std::fabs(std::cos(phase * 0.75)));
    payload.dy = -1.5f + static_cast<float>(3.0 * std::sin(phase * 0.25));

    std::cout << "[Vision::NVMM] Event tick=" << last_event_tick_
              << " frame=" << last_frame_id_
              << " object=" << payload.object_id
              << " confidence=" << payload.confidence
              << (using_synthetic_ingest_ ? " [synthetic_ingress]" : " [upstream_ingress]")
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

} // namespace vision
