cmake_minimum_required(VERSION 3.24)
project(durakk_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 Ti = Compute Capability 8.9 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Оптимизация
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

# CUDA runtime
find_package(CUDAToolkit REQUIRED)

# Исходники
set(CXX_SOURCES
    src/main.cpp
    src/card.cpp
    src/deck.cpp
    src/game_state.cpp
    src/rules.cpp
    src/bot.cpp
    src/protocol.cpp
    src/mcts.cpp
    src/gpu_engine.cpp
)

set(CUDA_SOURCES
    src/gpu_kernels.cu
    src/gpu_launchers.cu
)

# Единый исполняемый файл
add_executable(durakk_engine ${CXX_SOURCES} ${CUDA_SOURCES})

target_include_directories(durakk_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(durakk_engine PRIVATE
    CUDA::cudart
    CUDA::curand
)

# Копируем в корень и dist
add_custom_command(TARGET durakk_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:durakk_engine>
            ${CMAKE_SOURCE_DIR}/../durakk_engine.exe
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:durakk_engine>
            ${CMAKE_SOURCE_DIR}/dist/durakk_engine.exe
)

# Для отладки: выводим инфо о CUDA
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "GPU Target: RTX 4060 Ti (SM 8.9, 8GB VRAM, 34 SMs)")
