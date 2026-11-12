cmake_minimum_required(VERSION 3.20)
project(gemma_amx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    -O3
    -mcpu=native
    -flto
    -fno-math-errno
    -funroll-loops
    -DACCELERATE_NEW_LAPACK
    -Wall
    -Wextra
    -Wno-unused-parameter
)

include_directories(
    src
    third_party
    third_party/httplib
    third_party/nlohmann
    third_party/minja
)

set(SOURCES
    src/gguf/gguf.cpp
    src/gguf/quant.cpp
    src/tokenizer/tokenizer.cpp
    src/engine/gemma4.cpp
    src/server/main.cpp
)

add_executable(gemma-amx ${SOURCES})

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)

target_link_libraries(gemma-amx PRIVATE
    ${ACCELERATE_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
    pthread
)

# Test target
add_executable(test_amx
    tests/test_amx.cpp
    src/gguf/gguf.cpp
    src/gguf/quant.cpp
    src/tokenizer/tokenizer.cpp
    src/engine/gemma4.cpp
)
target_link_libraries(test_amx PRIVATE
    ${ACCELERATE_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
    pthread
)
