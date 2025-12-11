add_library(hexagon_utils STATIC
    errorMetrics.cpp
    loadTensor.cpp
    helperFunction.cpp
)

target_include_directories(hexagon_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})