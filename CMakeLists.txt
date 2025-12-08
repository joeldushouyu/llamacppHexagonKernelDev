cmake_minimum_required(VERSION 3.21)

project(ggml-hexagon-test LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Add the ggml library from llama.cpp
# This will build ggml as a subdirectory
set(GGML_STANDALONE OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
set(GGML_HEXAGON_HTP_DEBUG ON CACHE BOOL "" FORCE)  # Enable DSP-side debug logs
set(GGML_OPENCL ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CURL OFF CACHE BOOL "" FORCE)

# Add ggml subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

# Add utils library
add_library(hexagon_utils STATIC
    utils/errorMetrics.cpp
    utils/errorMetrics.hpp
    utils/loadTensor.cpp
    utils/loadTensor.hpp
)

# Include directories for utils library
target_include_directories(hexagon_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/src
)

# Link ggml to utils library
target_link_libraries(hexagon_utils PUBLIC
    ggml
)

# Add executable
add_executable(hexagon_test
    test_hexagon.cpp
)

# Include directories
target_include_directories(hexagon_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Link libraries - ggml, ggml-hexagon, and hexagon_utils
target_link_libraries(hexagon_test PRIVATE
    ggml
    ggml-hexagon
    hexagon_utils
)

# Set output directory
set_target_properties(hexagon_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
