#include "errorMetrics.hpp"



ErrorMetrics calculate_error_metrics(const float* reference, const float* computed, int n_elements) {
    ErrorMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f};
    
    double l1_error = 0.0;
    double l2_error = 0.0;
    double l1_norm_ref = 0.0;
    double l2_norm_ref = 0.0;
    double dot_product = 0.0;
    double norm_ref = 0.0;
    double norm_computed = 0.0;
    double squared_error = 0.0;
    
    for (int i = 0; i < n_elements; i++) {
        double diff = fabs(computed[i] - reference[i]);
        double ref_abs = fabs(reference[i]);
        
        l1_error += diff;
        l2_error += diff * diff;
        l1_norm_ref += ref_abs;
        l2_norm_ref += reference[i] * reference[i];
        
        dot_product += reference[i] * computed[i];
        norm_ref += reference[i] * reference[i];
        norm_computed += computed[i] * computed[i];
        
        squared_error += diff * diff;
    }
    
    // L1 relative error
    metrics.l1_relative_error = (l1_norm_ref > 1e-10) ? (l1_error / l1_norm_ref) : 0.0f;
    
    // L2 relative error
    metrics.l2_relative_error = (l2_norm_ref > 1e-10) ? sqrt(l2_error / l2_norm_ref) : 0.0f;
    
    // Cosine similarity
    double denom = sqrt(norm_ref) * sqrt(norm_computed);
    metrics.cosine_similarity = (denom > 1e-10) ? (dot_product / denom) : 0.0f;
    
    // RMS error
    metrics.rms_error = sqrt(squared_error / n_elements);
    
    return metrics;
}

void print_error_metrics(const ErrorMetrics& metrics) {
    GGML_LOG_INFO("\n--- Error Metrics ---\n");
    GGML_LOG_INFO("L1 Relative Error:    %.6f\n", metrics.l1_relative_error);
    GGML_LOG_INFO("L2 Relative Error:    %.6f\n", metrics.l2_relative_error);
    GGML_LOG_INFO("Cosine Similarity:    %.8f\n", metrics.cosine_similarity);
    GGML_LOG_INFO("RMS Error:            %.6e\n", metrics.rms_error);
    GGML_LOG_INFO("--------------------\n");
}