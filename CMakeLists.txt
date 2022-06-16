cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(csd LANGUAGES CUDA CXX)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

#set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include ${TORCH_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
add_subdirectory(lib)
add_subdirectory(demos)

target_link_libraries(benchmark_csd modules nets ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(train_vanilla_locvol modules nets training simulators ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(price_vanilla_locvol modules nets training simulators ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(grad_check modules ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(layers_test modules nets simulators ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
