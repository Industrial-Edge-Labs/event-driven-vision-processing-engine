#include <iostream>
#include <signal.h>
#include <cstdlib>
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

    // Initialize the NVMM orchestrator layer for 1080p, assuming standard 3-channel buffers
    vision::ZeroCopyIngest ingest(1920, 1080, 3);
    
    if (!ingest.allocateMemory()) {
        std::cerr << "[VisionEngine] Critical Error: Failed to allocate pinned contiguous NVMM bounds.\n";
        return EXIT_FAILURE;
    }

    // High throughput loop
    uint64_t frame_index = 0;
    while (is_running) {
        // Poll NIC stream buffer bypassing CPU caching
        if (ingest.pollNetworkInterface()) {
            ingest.processTemporalDerivative(frame_index);

            if (ingest.hasThresholdEvent()) {
                // If derivative |dI/dt| exceeds threshold, trigger ZMQ dispatch to the Decision System
                // This acts as a sparse topological forward mapping
                ingest.dispatchInferenceEvent();
            }
        }
        
        frame_index++;
    }

    ingest.releaseMemory();
    std::cout << "[VisionEngine] Pipeline decoupled securely.\n";
    return EXIT_SUCCESS;
}
