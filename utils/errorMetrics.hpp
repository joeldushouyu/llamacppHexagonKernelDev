#ifndef __ERROR_METRICS_HPP__
#define __ERROR_METRICS_HPP__
#include <time.h>
#include <math.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

// Error metric calculation
struct ErrorMetrics {
    float l1_relative_error;
    float l2_relative_error;
    float cosine_similarity;
    float rms_error;
};







ErrorMetrics calculate_error_metrics(const float* reference, const float* computed, int n_elements) ;

void print_error_metrics(const ErrorMetrics& metrics);
#endif 