#pragma once

#include <cstdint>
#include <atomic>
#include <vector>

namespace vision {

/**
 * Encapsulates the hardware abstraction layer bridging the Network Interface Card (NIC)
 * directly into the NVIDIA GPU Memory Management (NVMM) bypassing Host RAM.
 * Ensures the PCI-e throughput operates at zero-copy latency targets.
 */
class ZeroCopyIngest {
public:
    ZeroCopyIngest(uint32_t width_px, uint32_t height_px, uint32_t channels);
    ~ZeroCopyIngest();

    // Disable copy semantics
    ZeroCopyIngest(const ZeroCopyIngest&) = delete;
    ZeroCopyIngest& operator=(const ZeroCopyIngest&) = delete;

    bool allocateMemory();
    void releaseMemory();

    bool pollNetworkInterface();
    void processTemporalDerivative(uint64_t clock_tick);
    bool hasThresholdEvent() const;
    void dispatchInferenceEvent(uint64_t clock_tick);

private:
    void* zmq_context_;
    void* zmq_publisher_;
    void* zmq_subscriber_;
    
    uint32_t w_;
    uint32_t h_;
    uint32_t c_;
    
    // Pinned device pointers
    void* d_frame_current_{nullptr};
    void* d_frame_previous_{nullptr};
    void* d_flow_magnitude_{nullptr};

    std::atomic<bool> event_triggered_{false};
    std::atomic<uint32_t> next_object_id_{1};
    uint64_t last_frame_id_{0};
    uint64_t synthetic_frame_cursor_{0};
    uint64_t last_event_tick_{0};
    float last_event_score_{0.0f};
    bool using_synthetic_ingest_{true};
    float detection_threshold_{0.85f}; // Example tensor activation probability minimum
};

} // namespace vision
