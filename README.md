# Event-Driven Vision Processing Engine

This repository contains the high-performance, event-driven computer vision layer of the Industrial Edge ecosystem. It relies on the principle of transforming raw contiguous memory spaces (pixels) into structured, sparse spatial events.

## System Architecture

The vision engine avoids operating on redundant data by employing:
1. NVMM (NVIDIA Memory Management) Zero-Copy ingestion to prevent PCI-e bus bottlenecks between the NIC and the GPU.
2. TensorRT for INT8/FP16 quantized inference of deep architectures.
3. CUDA Streams for asynchronous gradient map computation and background segmentation.

## Build Instructions (Linux / Windows)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Dependencies
- NVIDIA CUDA Toolkit 12.x+
- TensorRT 8.6+
- OpenCV 4.x (compiled with CUDA support)
- ZeroMQ (libzmq)

## Mathematical Paradigm

Rather than processing $I(x,y,t)$ at every frame $t$, the system computes the temporal derivative $\frac{\partial I}{\partial t}$ using hardware-accelerated optical flow and background subtraction matrices. Only when $||\frac{\partial I}{\partial t}|| > \epsilon$ does the deep learning inference graph execute, ensuring $O(1)$ thermal and memory ceilings.
