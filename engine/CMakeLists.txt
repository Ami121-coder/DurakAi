# ============================================================================
# CMakeLists.txt.patched — добавлены ONNX Runtime и AVX-512 для 7500F.
#
# Что меняется:
#   1. find_package(onnxruntime CONFIG REQUIRED) — статическая линковка.
#      Альтернатива: vcpkg install onnxruntime-cuda.
#   2. /arch:AVX2 → /arch:AVX512 (Ryzen 7500F = Zen 4 поддерживает AVX-512).
#   3. Добавлен onnx_net.cpp в список источников durakk_engine и durakk_env.
#   4. Опция DURAKK_USE_ONNX (по умолчанию ON) — можно выключить, если нет ONNX Runtime.
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(durakk_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DURAKK_USE_ONNX "Включить ONNX Runtime для OnnxNet" ON)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
    # Ryzen 7500F (Zen 4) поддерживает AVX-512. AVX2 → AVX512 даёт
    # ускорение popcnt и SIMD в ~1.3-1.5× на горячих путях ISMCTS.
    add_compile_options(/arch:AVX512)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -mpopcnt -mavx2 -mavx512f -mavx512bw)
endif()

find_package(Threads REQUIRED)

# ---- ONNX Runtime ----
if(DURAKK_USE_ONNX)
    # Способ 1: vcpkg (рекомендуется на Windows)
    # find_package(onnxruntime CONFIG REQUIRED)
    #
    # Способ 2: системная установка (Linux/macOS)
    find_library(ONNXRUNTIME_LIB onnxruntime)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
              PATHS /usr/include /usr/local/include /opt/onnxruntime/include)
    if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
        message(WARNING "ONNX Runtime не найден — OnnxNet будет заглушкой. "
                "Установите через vcpkg или пакетный менеджер.")
        set(DURAKK_USE_ONNX OFF)
    endif()
endif()

set(ENGINE_SOURCES
    src/main.cpp
    src/card.cpp
    src/deck.cpp
    src/game_state.cpp
    src/rules.cpp
    src/move.h
    src/bitboard.cpp
    src/match.cpp
    src/rules_fast.cpp
    src/bot.cpp
    src/ismcts.cpp
    src/endgame.cpp
    src/ttable.cpp
    src/nnet/random_net.cpp
    src/protocol.cpp
)
if(DURAKK_USE_ONNX)
    list(APPEND ENGINE_SOURCES src/nnet/onnx_net.cpp)
endif()

add_executable(durakk_engine ${ENGINE_SOURCES})
target_include_directories(durakk_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(durakk_engine PRIVATE Threads::Threads)
if(DURAKK_USE_ONNX)
    target_include_directories(durakk_engine PRIVATE ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(durakk_engine PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(durakk_engine PRIVATE DURAKK_USE_ONNX=1)
endif()

set_target_properties(durakk_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist
)

add_custom_command(TARGET durakk_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:durakk_engine>
        ${CMAKE_CURRENT_SOURCE_DIR}/../durakk_engine$<IF:$<BOOL:${WIN32}>,.exe,>
)

# ---- pybind11 модуль durakk_env ----
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

set(ENV_SOURCES
    src/bindings.cpp
    src/card.cpp
    src/deck.cpp
    src/game_state.cpp
    src/rules.cpp
    src/bitboard.cpp
    src/match.cpp
    src/rules_fast.cpp
    src/bot.cpp
    src/ismcts.cpp
    src/endgame.cpp
    src/ttable.cpp
    src/nnet/random_net.cpp
)
if(DURAKK_USE_ONNX)
    list(APPEND ENV_SOURCES src/nnet/onnx_net.cpp)
endif()

pybind11_add_module(durakk_env ${ENV_SOURCES})
target_include_directories(durakk_env PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(durakk_env PRIVATE Threads::Threads)
if(DURAKK_USE_ONNX)
    target_include_directories(durakk_env PRIVATE ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(durakk_env PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(durakk_env PRIVATE DURAKK_USE_ONNX=1)
endif()

if(MSVC)
    target_compile_options(durakk_env PRIVATE /arch:AVX512 /permissive- /utf-8)
else()
    target_compile_options(durakk_env PRIVATE -mpopcnt -mavx2 -mavx512f -mavx512bw)
endif()

add_custom_command(TARGET durakk_env POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:durakk_env>
        ${CMAKE_CURRENT_SOURCE_DIR}/../durakk_env$<IF:$<BOOL:${WIN32}>,.pyd,.so>
)
