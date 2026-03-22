#include <cmath>
#include <iostream>

#include "vision/Contracts.hpp"
#include "vision/EventSynthesizer.hpp"

namespace {

bool test_frame_shape_validation() {
    const vision::UpstreamFrameEnvelope envelope{123, 5, 1920, 1080, 3};
    return vision::is_expected_frame_shape(envelope, 1920, 1080, 3)
        && !vision::is_expected_frame_shape(envelope, 1280, 720, 3);
}

bool test_temporal_score_bounds() {
    const float score = vision::compute_temporal_score(12, 44, false);
    return score >= 0.0f && score <= 0.999f;
}

bool test_synthesized_payload_is_valid() {
    const auto snapshot = vision::synthesize_event(16, 88, true, 7);
    return snapshot.event_tick == 16
        && snapshot.frame_id == 88
        && snapshot.using_fallback_ingest
        && snapshot.payload.object_id == 7
        && vision::is_valid_inference_payload(snapshot.payload);
}

} // namespace

int main() {
    if (!test_frame_shape_validation()) {
        std::cerr << "test_frame_shape_validation failed\n";
        return 1;
    }
    if (!test_temporal_score_bounds()) {
        std::cerr << "test_temporal_score_bounds failed\n";
        return 1;
    }
    if (!test_synthesized_payload_is_valid()) {
        std::cerr << "test_synthesized_payload_is_valid failed\n";
        return 1;
    }

    std::cout << "vision_engine_tests passed\n";
    return 0;
}
