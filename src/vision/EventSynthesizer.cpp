#include "EventSynthesizer.hpp"

#include <chrono>
#include <cmath>

namespace vision {

uint64_t monotonic_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count());
}

float compute_temporal_score(uint64_t clock_tick, uint64_t frame_id, bool using_fallback_ingest) {
    const double phase = static_cast<double>((clock_tick + frame_id) % 720) * 0.021;
    const float temporal_energy = static_cast<float>(0.45 + 0.35 * std::sin(phase));
    const float transport_bias = using_fallback_ingest ? 0.02f : 0.08f;
    const float edge_density = static_cast<float>(0.06 + 0.06 * std::cos(phase * 0.5));
    return clamp_unit(temporal_energy + transport_bias + edge_density);
}

TemporalEventSnapshot synthesize_event(
    uint64_t clock_tick,
    uint64_t frame_id,
    bool using_fallback_ingest,
    uint32_t object_id
) {
    TemporalEventSnapshot snapshot{};
    snapshot.event_tick = clock_tick;
    snapshot.frame_id = frame_id;
    snapshot.score = compute_temporal_score(clock_tick, frame_id, using_fallback_ingest);
    snapshot.using_fallback_ingest = using_fallback_ingest;

    const double phase = static_cast<double>((clock_tick + frame_id) % 360) * 0.017453292519943295;
    snapshot.payload.timestamp = monotonic_now_ns();
    snapshot.payload.object_id = object_id;
    snapshot.payload.confidence = clamp_unit(0.55f + (snapshot.score * 0.35f));
    snapshot.payload.x = 960.0f + static_cast<float>(420.0 * std::sin(phase));
    snapshot.payload.y = 540.0f + static_cast<float>(180.0 * std::cos(phase * 0.5));
    snapshot.payload.dx = 4.0f + static_cast<float>(18.0 * std::fabs(std::cos(phase * 0.75)));
    snapshot.payload.dy = -1.5f + static_cast<float>(3.0 * std::sin(phase * 0.25));

    return snapshot;
}

} // namespace vision
