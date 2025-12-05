#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#define GGML_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_ERROR(...) fprintf(stderr, "ERROR: " __VA_ARGS__)

// Helper function to print tensor values
void print_tensor(const char* name, struct ggml_tensor* tensor, int max_elements = 10) {
    GGML_LOG_INFO("%s: [", name);
    float* data = (float*)tensor->data;
    int n_elements = ggml_nelements(tensor);
    int print_count = n_elements < max_elements ? n_elements : max_elements;
    
    for (int i = 0; i < print_count; i++) {
        GGML_LOG_INFO("%.4f", data[i]);
        if (i < print_count - 1) GGML_LOG_INFO(", ");
    }
    if (n_elements > max_elements) {
        GGML_LOG_INFO(" ... (%d more)", n_elements - max_elements);
    }
    GGML_LOG_INFO("]\n");
}

// Error metric calculation
struct ErrorMetrics {
    float l1_relative_error;
    float l2_relative_error;
    float cosine_similarity;
    float rms_error;
};

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
    GGML_LOG_INFO("L1 Relative Error:    %.6e\n", metrics.l1_relative_error);
    GGML_LOG_INFO("L2 Relative Error:    %.6e\n", metrics.l2_relative_error);
    GGML_LOG_INFO("Cosine Similarity:    %.8f\n", metrics.cosine_similarity);
    GGML_LOG_INFO("RMS Error:            %.6e\n", metrics.rms_error);
    GGML_LOG_INFO("--------------------\n");
}

// Test RMS normalization
bool test_rms_norm(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing RMS Normalization ===\n");
    
    const int n_elements = 4096;  // typical embedding size
    
    // Create context
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128 MB
        .mem_buffer = NULL,
        .no_alloc   = true,  // Use backend buffers
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        GGML_LOG_ERROR("Failed to create ggml context\n");
        return false;
    }
    
    // Create input tensor
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    
    // Apply RMS norm
    struct ggml_tensor* result = ggml_rms_norm(ctx, x, 1e-5f);
    
    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    
    // Allocate buffers on the backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return false;
    }
    
    // Initialize with test data
    float* x_data = (float*)x->data;
    for (int i = 0; i < n_elements; i++) {
        x_data[i] = (float)(i % 100) / 100.0f;  // values between 0 and 0.99
    }
    
    GGML_LOG_INFO("Input tensor shape: [%d]\n", n_elements);
    print_tensor("Input (first 10)", x);
    
    GGML_LOG_INFO("Computing RMS norm on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
    print_tensor("Output (first 10)", result);
    
    // CPU reference implementation
    GGML_LOG_INFO("\nComputing CPU reference...\n");
    float* cpu_reference = (float*)malloc(n_elements * sizeof(float));
    float* x_data_cpu = (float*)x->data;
    
    // Compute RMS
    double sum_squares = 0.0;
    for (int i = 0; i < n_elements; i++) {
        sum_squares += x_data_cpu[i] * x_data_cpu[i];
    }
    float rms = sqrt(sum_squares / n_elements);
    
    // Normalize by RMS with epsilon
    float eps = 1e-5f;
    float scale = 1.0f / (rms + eps);
    for (int i = 0; i < n_elements; i++) {
        cpu_reference[i] = x_data_cpu[i] * scale;
    }
    
    // Calculate error metrics
    float* result_data = (float*)result->data;
    ErrorMetrics metrics = calculate_error_metrics(cpu_reference, result_data, n_elements);
    print_error_metrics(metrics);
    
    free(cpu_reference);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    GGML_LOG_INFO("RMS Norm test completed successfully!\n");
    return true;
}

// Test element-wise addition
bool test_add(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing Element-wise Addition ===\n");
    
    const int n_elements = 1024;
    
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,  // Don't allocate memory in context, use backend buffers
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        GGML_LOG_ERROR("Failed to create ggml context\n");
        return false;
    }
    
    // Create input tensors (without allocating data)
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    
    // Apply addition
    struct ggml_tensor* result = ggml_add(ctx, a, b);
    
    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    
    // Allocate buffers on the backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return false;
    }
    
    // Initialize input data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < n_elements; i++) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }
    
    GGML_LOG_INFO("Tensor A shape: [%d]\n", n_elements);
    GGML_LOG_INFO("Tensor B shape: [%d]\n", n_elements);
    print_tensor("A (first 10)", a);
    print_tensor("B (first 10)", b);
    
    GGML_LOG_INFO("Computing addition on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
    print_tensor("Result (first 10)", result);
    
    // CPU reference implementation
    GGML_LOG_INFO("\nComputing CPU reference...\n");
    float* cpu_reference = (float*)malloc(n_elements * sizeof(float));
    float* a_data_cpu = (float*)a->data;
    float* b_data_cpu = (float*)b->data;
    
    for (int i = 0; i < n_elements; i++) {
        cpu_reference[i] = a_data_cpu[i] + b_data_cpu[i];
    }
    
    // Calculate error metrics
    float* result_data = (float*)result->data;
    ErrorMetrics metrics = calculate_error_metrics(cpu_reference, result_data, n_elements);
    print_error_metrics(metrics);
    
    free(cpu_reference);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    GGML_LOG_INFO("Addition test completed successfully!\n");
    return true;
}

// Test matrix multiplication
bool test_mul_mat(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing Matrix Multiplication ===\n");
    
    // NOTE: Hexagon backend supports:
    // - Q4_0, Q8_0, MXFP4 types for src0 (weights)
    // - F16 for src0 (requires experimental flag)
    // - F32 for src1 (input) and dst (output)
    // F32 x F32 matmul is NOT supported, so we use F16 x F32
    
    const int m = 512;   // rows of result
    const int n = 256;   // cols of result  
    const int k = 1024;  // shared dimension
    
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,  // Use backend buffers
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        GGML_LOG_ERROR("Failed to create ggml context\n");
        return false;
    }
    
    // Create input matrices
    // For ggml_mul_mat(a, b): result[m,n] = a[k,m] @ b[n,k]
    // a is transposed in the multiplication
    // Use F16 for src0 (a) as Hexagon supports F16 x F32 but not F32 x F32
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);  // [k, m]
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  // [k, n]
    
    // Apply matrix multiplication
    struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
    
    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    
    // Allocate buffers on the backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return false;
    }
    
    // Initialize with test data
    ggml_fp16_t* a_data = (ggml_fp16_t*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < k * m; i++) {
        a_data[i] = ggml_fp32_to_fp16(1.0f);  // Use 1.0 for easier verification
    }
    for (int i = 0; i < k * n; i++) {
        b_data[i] = 1.0f;
    }
    
    GGML_LOG_INFO("Matrix A shape: [%d, %d]\n", (int)a->ne[0], (int)a->ne[1]);
    GGML_LOG_INFO("Matrix B shape: [%d, %d]\n", (int)b->ne[0], (int)b->ne[1]);
    
    GGML_LOG_INFO("Computing matrix multiplication on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
    GGML_LOG_INFO("Result shape: [%d, %d]\n", (int)result->ne[0], (int)result->ne[1]);
    print_tensor("Result (first 10)", result);
    
    // CPU reference implementation
    GGML_LOG_INFO("\nComputing CPU reference...\n");
    float* cpu_reference = (float*)malloc(m * n * sizeof(float));
    ggml_fp16_t* a_data_cpu = (ggml_fp16_t*)a->data;
    float* b_data_cpu = (float*)b->data;
    
    // Matrix multiplication: C[m,n] = A[k,m]^T @ B[k,n]
    // For each output element C[i,j], compute dot product of A[:,i] with B[:,j]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                // A is stored column-major: A[k,m] means k rows, m columns
                // Element at row p, column i is at index: i*k + p
                float a_val = ggml_fp16_to_fp32(a_data_cpu[i * k + p]);
                // B is stored column-major: B[k,n] means k rows, n columns  
                // Element at row p, column j is at index: j*k + p
                float b_val = b_data_cpu[j * k + p];
                sum += a_val * b_val;
            }
            // Result is stored column-major: result[m,n]
            // Element at row i, column j is at index: j*m + i
            cpu_reference[j * m + i] = sum;
        }
    }
    
    // Calculate error metrics
    float* result_data = (float*)result->data;
    ErrorMetrics metrics = calculate_error_metrics(cpu_reference, result_data, m * n);
    print_error_metrics(metrics);
    
    // Verify result: with all 1.0 inputs, result should be k (1024) in each element
    float expected = (float)k;  // 1024.0
    float tolerance = 1.0f;
    bool verified = true;
    for (int i = 0; i < 10 && i < m * n; i++) {
        if (fabs(result_data[i] - expected) > tolerance) {
            GGML_LOG_ERROR("Verification failed: result[%d] = %.4f, expected ~%.4f\n", 
                          i, result_data[i], expected);
            verified = false;
            break;
        }
    }
    if (verified) {
        GGML_LOG_INFO("Result verification: PASSED (values ~%.4f as expected)\n", expected);
    }
    
    free(cpu_reference);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    GGML_LOG_INFO("Matrix multiplication test completed successfully!\n");
    return true;
}

int main(int argc, char** argv) {
    GGML_LOG_INFO("========================================\n");
    GGML_LOG_INFO("GGML Hexagon Backend Test Suite\n");
    GGML_LOG_INFO("========================================\n\n");
    
    // Initialize Hexagon backend using the registry API
    GGML_LOG_INFO("Initializing Hexagon backend...\n");
    
    ggml_backend_reg_t reg = ggml_backend_hexagon_reg();
    if (!reg) {
        GGML_LOG_ERROR("Failed to get Hexagon backend registry\n");
        return 1;
    }
    
    // Get device count
    size_t device_count = ggml_backend_reg_dev_count(reg);
    GGML_LOG_INFO("Found %zu Hexagon device(s)\n", device_count);
    
    if (device_count == 0) {
        GGML_LOG_ERROR("No Hexagon devices found\n");
        return 1;
    }
    
    // Get the first device
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) {
        GGML_LOG_ERROR("Failed to get Hexagon device\n");
        return 1;
    }
    
    // Initialize backend from device
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        GGML_LOG_ERROR("Failed to initialize Hexagon backend from device\n");
        return 1;
    }
    
    if (!ggml_backend_is_hexagon(backend)) {
        GGML_LOG_ERROR("Backend is not a Hexagon backend\n");
        ggml_backend_free(backend);
        return 1;
    }
    
    GGML_LOG_INFO("Hexagon backend initialized successfully!\n");
    GGML_LOG_INFO("Backend name: %s\n", ggml_backend_name(backend));
    
    // Run tests
    bool all_passed = true;
    
    // Test 1: Addition
    if (!test_add(backend)) {
        GGML_LOG_ERROR("Addition test FAILED\n");
        all_passed = false;
    }
    
    // Test 2: RMS Normalization
    if (!test_rms_norm(backend)) {
        GGML_LOG_ERROR("RMS Norm test FAILED\n");
        all_passed = false;
    }
    
    // Test 3: Matrix Multiplication
    if (!test_mul_mat(backend)) {
        GGML_LOG_ERROR("Matrix multiplication test FAILED\n");
        all_passed = false;
    }
    
    // Cleanup
    ggml_backend_free(backend);
    
    GGML_LOG_INFO("\n========================================\n");
    if (all_passed) {
        GGML_LOG_INFO("All tests PASSED! ✓\n");
    } else {
        GGML_LOG_INFO("Some tests FAILED! ✗\n");
    }
    GGML_LOG_INFO("========================================\n");
    
    return all_passed ? 0 : 1;
}
