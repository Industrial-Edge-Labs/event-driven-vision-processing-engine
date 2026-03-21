# Event-Driven Vision Processing Engine

This repository contains the low-latency vision runtime of the Industrial Edge ecosystem. Its job is to convert dense video ingress into sparse, structured motion events that downstream deterministic systems can evaluate quickly.

## What It Does Now

- Accepts upstream frame envelopes from the video stream orchestrator on `tcp://127.0.0.1:6000` when available.
- Falls back to synthetic ingress when the upstream producer is offline, so the module stays debuggable in isolation.
- Computes a lightweight temporal-energy score that approximates `|dI/dt|` behavior and only emits events when the configured threshold is exceeded.
- Publishes compact binary inference events to the decision layer on `tcp://127.0.0.1:5555`.
- Supports dry-run execution through `VISION_ENGINE_MAX_FRAMES` or a positional CLI argument.

## Build Modes

The code is intentionally buildable in a lightweight mock mode, because the current implementation does not require CUDA or OpenCV to run its synthetic path.

### Minimal local build

```bash
cmake -S . -B build -DVISION_ENGINE_ENABLE_ZEROMQ=OFF
cmake --build build
```

### ZeroMQ-enabled build

```bash
cmake -S . -B build
cmake --build build
```

### Optional CUDA/OpenCV linkage

```bash
cmake -S . -B build \
  -DVISION_ENGINE_ENABLE_ZEROMQ=ON \
  -DVISION_ENGINE_ENABLE_CUDA=ON \
  -DVISION_ENGINE_ENABLE_OPENCV=ON
cmake --build build
```

## Runtime

```bash
./vision_engine 500
```

Or:

```bash
VISION_ENGINE_MAX_FRAMES=500 ./vision_engine
```

## Event Contract

The emitted binary payload contains:

1. `timestamp` in nanoseconds.
2. `object_id` as a monotonically increasing identifier.
3. `confidence` in the `[0, 1)` range.
4. `x`, `y`, `dx`, `dy` as sparse motion descriptors.

This keeps the module aligned with the decision engine without introducing JSON parsing into the hot path.

## Notes

- If ZeroMQ is unavailable, the engine keeps generating and logging events locally.
- The current temporal derivative is still a deterministic mock, but now it varies over time and reflects whether ingress is synthetic or upstream-fed.
- The detailed architecture write-up for this node lives in `docs-Industrial-Edge-Labs/event-driven-vision-processing-engine/architecture.md`.
