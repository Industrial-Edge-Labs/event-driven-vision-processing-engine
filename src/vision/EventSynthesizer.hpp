#pragma once

#include <cstdint>

#include "Contracts.hpp"

namespace vision {

struct TemporalEventSnapshot {
    uint64_t event_tick;
    uint64_t frame_id;
    float score;
    bool using_fallback_ingest;
    InferencePayload payload;
};

uint64_t monotonic_now_ns();
float compute_temporal_score(uint64_t clock_tick, uint64_t frame_id, bool using_fallback_ingest);
TemporalEventSnapshot synthesize_event(
    uint64_t clock_tick,
    uint64_t frame_id,
    bool using_fallback_ingest,
    uint32_t object_id
);

} // namespace vision
