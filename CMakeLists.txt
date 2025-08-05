cmake_minimum_required(VERSION 3.18)
project(cuda_chroma_onefile)

find_package(OpenCV REQUIRED)
enable_language(CUDA)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cuda_chroma main.cu)
target_link_libraries(cuda_chroma ${OpenCV_LIBS} cuda cudart)
