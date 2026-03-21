cmake_minimum_required(VERSION 3.18)
project(EventDrivenVisionEngine VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89) # Ampere, Lovelace

add_compile_options(-O3 -march=native -ffast-math)

# Locate dependencies
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core cudabgsegm cudaimgproc)
find_package(ZeroMQ REQUIRED)
# TensoRT placeholder (find_package(nvinfer REQUIRED))

add_executable(vision_engine
    src/main.cpp
    src/vision/ZeroCopyIngest.cpp
)

target_include_directories(vision_engine PRIVATE src)
target_link_libraries(vision_engine PRIVATE 
    zmq 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
)
