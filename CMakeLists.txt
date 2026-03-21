cmake_minimum_required(VERSION 3.18)
project(EventDrivenVisionEngine VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VISION_ENGINE_ENABLE_ZEROMQ "Enable ZeroMQ ingress/egress integration." OFF)
option(VISION_ENGINE_ENABLE_CUDA "Enable optional CUDA runtime linkage." OFF)
option(VISION_ENGINE_ENABLE_OPENCV "Enable optional OpenCV linkage." OFF)

add_executable(vision_engine
    src/main.cpp
    src/vision/ZeroCopyIngest.cpp
)

target_include_directories(vision_engine PRIVATE src)
target_compile_options(vision_engine PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
)
target_compile_definitions(vision_engine PRIVATE
    VISION_ENGINE_USE_ZMQ=$<BOOL:${VISION_ENGINE_ENABLE_ZEROMQ}>
    VISION_ENGINE_USE_CUDA=$<BOOL:${VISION_ENGINE_ENABLE_CUDA}>
    VISION_ENGINE_USE_OPENCV=$<BOOL:${VISION_ENGINE_ENABLE_OPENCV}>
)

if (VISION_ENGINE_ENABLE_ZEROMQ)
    find_package(ZeroMQ REQUIRED)
    target_link_libraries(vision_engine PRIVATE zmq)
endif()

if (VISION_ENGINE_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(vision_engine PRIVATE CUDA::cudart)
endif()

if (VISION_ENGINE_ENABLE_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
    target_link_libraries(vision_engine PRIVATE ${OpenCV_LIBS})
endif()
