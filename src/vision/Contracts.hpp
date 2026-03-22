#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace vision {

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
static_assert(sizeof(UpstreamFrameEnvelope) == 28, "UpstreamFrameEnvelope wire format changed unexpectedly.");

inline float clamp_unit(float value) {
    return std::clamp(value, 0.0f, 0.999f);
}

inline bool is_expected_frame_shape(
    const UpstreamFrameEnvelope& envelope,
    uint32_t width,
    uint32_t height,
    uint32_t channels
) {
    return envelope.width == width && envelope.height == height && envelope.channels == channels;
}

inline bool is_valid_inference_payload(const InferencePayload& payload) {
    return payload.timestamp > 0
        && payload.object_id > 0
        && std::isfinite(payload.confidence)
        && payload.confidence >= 0.0f
        && payload.confidence <= 0.999f
        && std::isfinite(payload.x)
        && std::isfinite(payload.y)
        && std::isfinite(payload.dx)
        && std::isfinite(payload.dy);
}

} // namespace vision
