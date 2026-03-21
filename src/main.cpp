#include <chrono>
#include <csignal>
#include <iostream>
#include <cstdlib>
#include <thread>
#include "vision/ZeroCopyIngest.hpp"

std::atomic<bool> is_running{true};

void interrupt_handler(int sig) {
    if (is_running.load()) {
        std::cout << "[VisionEngine] Halting deterministic loop synchronously on signal " << sig << "\n";
        is_running.store(false);
    }
}

int main(int argc, char** argv) {
    std::cout << "[VisionEngine] Initializing Zero-Copy DMA pipelines...\n";
    std::signal(SIGINT, interrupt_handler);
    uint64_t max_frames = 0;

    if (const char* env_frames = std::getenv("VISION_ENGINE_MAX_FRAMES")) {
        max_frames = std::strtoull(env_frames, nullptr, 10);
    }

    if (argc > 1) {
        max_frames = std::strtoull(argv[1], nullptr, 10);
    }

    // Initialize the NVMM orchestrator layer for 1080p, assuming standard 3-channel buffers
    vision::ZeroCopyIngest ingest(1920, 1080, 3);
    
    if (!ingest.allocateMemory()) {
        std::cerr << "[VisionEngine] Critical Error: Failed to allocate pinned contiguous NVMM bounds.\n";
        return EXIT_FAILURE;
    }

    // High throughput loop
    uint64_t frame_index = 0;
    uint64_t ingested_frames = 0;
    while (is_running.load(std::memory_order_acquire)) {
        // Poll NIC stream buffer bypassing CPU caching
        if (ingest.pollNetworkInterface()) {
            ingest.processTemporalDerivative(frame_index);
            ingested_frames++;

            if (ingest.hasThresholdEvent()) {
                // If derivative |dI/dt| exceeds threshold, trigger ZMQ dispatch to the Decision System
                // This acts as a sparse topological forward mapping
                ingest.dispatchInferenceEvent(frame_index);
            }

            if (max_frames > 0 && ingested_frames >= max_frames) {
                std::cout << "[VisionEngine] Dry-run target reached after " << ingested_frames
                          << " ingested frames.\n";
                is_running.store(false, std::memory_order_release);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(250));
        }
        
        frame_index++;
    }

    ingest.releaseMemory();
    std::cout << "[VisionEngine] Pipeline decoupled securely.\n";
    return EXIT_SUCCESS;
}
