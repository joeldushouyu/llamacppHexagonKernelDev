#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "loadTensor.hpp"
#include "helperFunction.hpp"
#include "errorMetrics.hpp"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"


// Test RMS normalization
bool test_rms_norm(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing RMS Normalization ===\n");
    
    const int n_elements = 4099;  // typical embedding size
    
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

    
    GGML_LOG_INFO("Computing RMS norm on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
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
    
    const int n_elements = 4307;
    
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
    
    // fix seed
    srand(42);
    // Initialize input data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < n_elements; i++) {
        a_data[i] = random_float(-5.0f, 5.0f);
        b_data[i] = random_float(-5.0f, 5.0f);
    }
    
    GGML_LOG_INFO("Tensor A shape: [%d]\n", n_elements);
    GGML_LOG_INFO("Tensor B shape: [%d]\n", n_elements);
    
    GGML_LOG_INFO("Computing addition on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
    
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




// Test element-wise multiplication
bool test_mul(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing Element-wise Multiplication ===\n");
    
    const int n_elements = 4307;
    
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
    
    // Apply multiplication
    struct ggml_tensor* result = ggml_mul(ctx, a, b);
    
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
    
    // fix seed
    srand(42);
    // Initialize input data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < n_elements; i++) {
        a_data[i] = random_float(-5.0f, 5.0f);
        b_data[i] = random_float(-5.0f, 5.0f);
    }
    
    GGML_LOG_INFO("Tensor A shape: [%d]\n", n_elements);
    GGML_LOG_INFO("Tensor B shape: [%d]\n", n_elements);
    
    GGML_LOG_INFO("Computing multiplication on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    

    // CPU reference implementation
    GGML_LOG_INFO("\nComputing CPU reference...\n");
    float* cpu_reference = (float*)malloc(n_elements * sizeof(float));
    float* a_data_cpu = (float*)a->data;
    float* b_data_cpu = (float*)b->data;
    
    for (int i = 0; i < n_elements; i++) {
        cpu_reference[i] = a_data_cpu[i] * b_data_cpu[i];
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



// Test matrix multiplication: F16 x F32
bool test_mul_mat_f16_f32(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing Matrix Multiplication (F16 x F32) ===\n");

    
    // const int m = 64;   // rows of result
    // const int n = 64;   // cols of result  
    // const int k = 64;  // shared dimension

    // const int m = 2048;   // rows of result
    // const int n = 1024;   // cols of result  
    // const int k = 512;  // shared dimension

    // const int m = 2050;   // rows of result
    // const int n = 1303;   // cols of result  
    // const int k = 1029;  // shared dimension


    
    // const int m = 4304;   // rows of result
    // const int n = 4096;   // cols of result  
    // const int k = 1152;  // shared dimension    
    
    // const int m = 1152;   // rows of result
    // const int n = 4096;   // cols of result  
    // const int k = 1152;  // shared dimension    

    // const int m = 1152;   // rows of result
    // const int n = 4096;   // cols of result  
    // const int k = 4304;  // shared dimension 
    
    
    // const int m = 1152;   // rows of result
    // const int n = 4096;   // cols of result  
    // const int k = 1152;  // shared dimension    


    // const int m = 1152;   // rows of result
    // const int n = 4096;   // cols of result  
    // const int k = 1152;  // shared dimension

    const int m = 1152;   // rows of result
    const int n = 4096;   // cols of result  
    const int k = 4304;  // shared dimension


    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = NULL,
        .no_alloc   = true,  // Use backend buffers
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        GGML_LOG_ERROR("Failed to create ggml context\n");
        return false;
    }
    
    // A is row major of mxk, B is column major of kxn
    // C is column major of mxn
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);  // [k, m]
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  // [k, n]
    
    

    // Apply matrix multiplication
    struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
    
    // print the ne and nb of a, b, and result
    GGML_LOG_INFO("Tensor A ne: [%d, %d], nb: [%d, %d]\n", (int)a->ne[0], (int)a->ne[1], (int)a->nb[0], (int)a->nb[1]);
    GGML_LOG_INFO("Tensor B ne: [%d, %d], nb: [%d, %d]\n", (int)b->ne[0], (int)b->ne[1], (int)b->nb[0], (int)b->nb[1]);
    GGML_LOG_INFO("Result Tensor ne: [%d, %d], nb: [%d, %d]\n", (int)result->ne[0], (int)result->ne[1], (int)result->nb[0], (int)result->nb[1]);

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
    
    // fix seed
    srand(42);
    ggml_fp16_t* a_data = (ggml_fp16_t*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < k * m; i++) {
        //a_data[i] = ggml_fp32_to_fp16(1.0f);
        // every 64 elemnts has same random value

        // if(i % 64 == 0) {
        //     rand_share_64 = random_float(-5.0f, 5.0f);
        // }
        // a_data[i] = ggml_fp32_to_fp16(rand_share_64);
        //a_data[i] =ggml_fp32_to_fp16(random_float(-5.0f, 5.0f));  // generate random value between -5 to 5


        //a_data[i] =   ggml_fp32_to_fp16( (i%1024)*1.0f);  // generate random value between -10 to 10
        a_data[i] =   ggml_fp32_to_fp16( random_float(-4, 4));  // generate random value between -4 to 4
    }
    for (int i = 0; i < k * n; i++) {
        //b_data[i] =(i%1024)*1.0f; //random_float(-4, 4);  // generate random value between -4 to 4

        b_data[i] = random_float(-4, 4); 
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
    


    // print first 10 value of result and cpu_reference for debug
    for(int i = 0; i < 10; i++) {
        GGML_LOG_INFO("Result[%d]: %f, CPU Reference[%d]: %f\n", i, result_data[i], i, cpu_reference[i]);
    }

    free(cpu_reference);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    GGML_LOG_INFO("F16 x F32 matrix multiplication test completed successfully!\n");
    return true;
}







// Test matrix multiplication: F16 x F32
bool test_mul_mat_f16_f32_with_input_A(ggml_backend_t backend, struct ggml_tensor* a_input) {
    GGML_LOG_INFO("\n=== Testing Matrix Multiplication (F16 x F32) ===\n");



    const int n = 4096;   // cols of result  



    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = NULL,
        .no_alloc   = true,  // Use backend buffers
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        GGML_LOG_ERROR("Failed to create ggml context\n");
        return false;
    }
    
    // A is row major of mxk, B is column major of kxn
    // C is column major of mxn
    // assert(a_input->ne[0] == k && a_input->ne[1] == m);
    int k = a_input->ne[0];
    int m = a_input->ne[1];
    assert(a_input->type == GGML_TYPE_F16);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  // [k, n]
    
    

    // Apply matrix multiplication
    struct ggml_tensor* result = ggml_mul_mat(ctx, a_input, b);
    
    // print the ne and nb of a, b, and result
    GGML_LOG_INFO("Tensor A ne: [%d, %d], nb: [%d, %d]\n", (int)a_input->ne[0], (int)a_input->ne[1], (int)a_input->nb[0], (int)a_input->nb[1]);
    GGML_LOG_INFO("Tensor B ne: [%d, %d], nb: [%d, %d]\n", (int)b->ne[0], (int)b->ne[1], (int)b->nb[0], (int)b->nb[1]);
    GGML_LOG_INFO("Result Tensor ne: [%d, %d], nb: [%d, %d]\n", (int)result->ne[0], (int)result->ne[1], (int)result->nb[0], (int)result->nb[1]);

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
    
    // fix seed
    srand(42);
    ggml_fp16_t* a_data = (ggml_fp16_t*)a_input->data;
    float* b_data = (float*)b->data;

    for (int i = 0; i < k * n; i++) {
        //b_data[i] =(i%1024)*1.0f; //random_float(-4, 4);  // generate random value between -4 to 4

        b_data[i] = random_float(-80, 80); 
    }
    
    GGML_LOG_INFO("Matrix A shape: [%d, %d]\n", (int)a_input->ne[0], (int)a_input->ne[1]);
    GGML_LOG_INFO("Matrix B shape: [%d, %d]\n", (int)b->ne[0], (int)b->ne[1]);
    
    // print first 10 value of A for debug
    for(int i = 0; i < 10; i++) {
        GGML_LOG_INFO("A[%d]: %f\n", i, ggml_fp16_to_fp32(a_data[i]));
    }


    GGML_LOG_INFO("Computing matrix multiplication on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
    GGML_LOG_INFO("Result shape: [%d, %d]\n", (int)result->ne[0], (int)result->ne[1]);
    
    // CPU reference implementation
    GGML_LOG_INFO("\nComputing CPU reference...\n");
    float* cpu_reference = (float*)malloc(m * n * sizeof(float));
    ggml_fp16_t* a_data_cpu = (ggml_fp16_t*)a_input->data;
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
    


    // print first 10 value of result and cpu_reference for debug
    for(int i = 0; i < 10; i++) {
        GGML_LOG_INFO("Result[%d]: %f, CPU Reference[%d]: %f\n", i, result_data[i], i, cpu_reference[i]);
    }

    free(cpu_reference);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    GGML_LOG_INFO("F16 x F32 matrix multiplication test completed successfully!\n");
    return true;

}



// void test_load_ggml_file(const std::string &filename) {
//     GGML_LOG_INFO("\n=== Loading GGUF File: %s ===\n", filename.c_str());

//     struct gguf_context * gguf_metadata_ctx = nullptr; 
//     struct ggml_context * ctx_gguf = nullptr; 

//     // Step 1: Load GGUF metadata without allocating tensor data
//     struct gguf_init_params params = {
//         .no_alloc = true,
//         .ctx = &ctx_gguf
//     };
//     gguf_metadata_ctx = gguf_init_from_file(filename.c_str(), params);
    
//     if (!gguf_metadata_ctx) {
//         GGML_LOG_ERROR("Failed to load GGUF file: %s\n", filename.c_str());
//         return;
//     }
    
//     GGML_LOG_INFO("GGUF file loaded successfully!\n");
    
//     // Step 2: Print general metadata
//     int n_kv = gguf_get_n_kv(gguf_metadata_ctx);
//     GGML_LOG_INFO("Number of key-value pairs: %d\n", n_kv);
    
//     // Print some key metadata (architecture, parameter count, etc.)
//     for (int i = 0; i < n_kv; i++) {
//         const char* key = gguf_get_key(gguf_metadata_ctx, i);
//         enum gguf_type type = gguf_get_kv_type(gguf_metadata_ctx, i);
        
//         // Print important metadata
//         if (type == GGUF_TYPE_STRING) {
//             const char* value = gguf_get_val_str(gguf_metadata_ctx, i);
//             GGML_LOG_INFO("  %s: %s\n", key, value);
//         } else if (type == GGUF_TYPE_UINT32) {
//             uint32_t value = gguf_get_val_u32(gguf_metadata_ctx, i);
//             GGML_LOG_INFO("  %s: %u\n", key, value);
//         } else if (type == GGUF_TYPE_FLOAT32) {
//             float value = gguf_get_val_f32(gguf_metadata_ctx, i);
//             GGML_LOG_INFO("  %s: %f\n", key, value);
//         }
//     }
    
//     // Step 3: Get tensor information
//     int n_tensors = gguf_get_n_tensors(gguf_metadata_ctx);
//     GGML_LOG_INFO("\nNumber of tensors: %d\n", n_tensors);
    
//     // Print tensor details
//     for (int i = 0; i < n_tensors; i++) {
//         const char* tensor_name = gguf_get_tensor_name(gguf_metadata_ctx, i);
//         struct ggml_tensor* tensor = ggml_get_tensor(ctx_gguf, tensor_name);
        
//         if (tensor) {
//             GGML_LOG_INFO("\nTensor %d: %s\n", i, tensor_name);
//             GGML_LOG_INFO("  Type: %s\n", ggml_type_name(tensor->type));
//             GGML_LOG_INFO("  Dimensions: [%lld", tensor->ne[0]);
//             for (int d = 1; d < GGML_MAX_DIMS && tensor->ne[d] > 1; d++) {
//                 GGML_LOG_INFO(", %lld", tensor->ne[d]);
//             }
//             GGML_LOG_INFO("]\n");
            
//             // Calculate total elements
//             size_t n_elements = 1;
//             for (int d = 0; d < GGML_MAX_DIMS; d++) {
//                 if (tensor->ne[d] > 1) n_elements *= tensor->ne[d];
//                 else break;
//             }
//             GGML_LOG_INFO("  Total elements: %zu\n", n_elements);
//             GGML_LOG_INFO("  Size: %zu bytes\n", ggml_nbytes(tensor));
//         }
//     }
    
//     // Step 4: To actually load tensor data, you need to allocate buffers
//     // Option A: Allocate on backend (Hexagon)
//     // ggml_backend_t backend = ...; // your backend
//     // ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_gguf, backend);
    
//     // Option B: Allocate on CPU
//     // You would need to re-initialize with no_alloc = false or manually allocate
    
//     // For now, we're just inspecting metadata
//     GGML_LOG_INFO("\n=== GGUF File Inspection Complete ===\n");
    
//     // Cleanup
//     gguf_free(gguf_metadata_ctx);
//     ggml_free(ctx_gguf);
// }




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
    
    // // // Test 1: Addition
    // if (!test_add(backend)) {
    //     GGML_LOG_ERROR("Addition test FAILED\n");
    //     all_passed = false;
    // }
   
    
    // if (!test_mul(backend)) {
    //     GGML_LOG_ERROR("Multiplication test FAILED\n");
    //     all_passed = false;
    // }    
    // // Test 2: RMS Normalization
    // if (!test_rms_norm(backend)) {
    //     GGML_LOG_ERROR("RMS Norm test FAILED\n");
    //     all_passed = false;
    // }
    
    // Test 3: Matrix Multiplication (F16 x F32)
    // if (!test_mul_mat_f16_f32(backend)) {
    //     GGML_LOG_ERROR("F16 x F32 matrix multiplication test FAILED\n");
    //     all_passed = false;
    // }
    
    // Test 4: Load GGUF file (if provided as argument)
    if (argc > 1) {


        struct gguf_context * gguf_ctx = nullptr; 
        struct ggml_context * ctx_gguf = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::unordered_map<std::string, struct ggml_tensor*> tensor_map;
        bool loaded = load_gguf_with_data(argv[1], backend,  
            gguf_ctx,
            ctx_gguf,
            buffer,
            tensor_map);
        if (!loaded) {
            GGML_LOG_ERROR("Failed to load GGUF file: %s\n", argv[1]);
            all_passed = false;
        } else {
            GGML_LOG_INFO("GGUF file loaded successfully: %s\n", argv[1]);
        }
        // // print all tensor names loaded
        // GGML_LOG_INFO("\n=== Loaded Tensors ===\n");
        // for (const auto& pair : tensor_map) {
        //     GGML_LOG_INFO("Tensor Name: %s\n", pair.first.c_str());     
        // }
        // release the gguf context and ggml context and buffer

        if (!test_mul_mat_f16_f32_with_input_A(backend, tensor_map["v.blk.0.ffn_down.weight"])) {
    
            all_passed = false;
        }        


        if (!test_mul_mat_f16_f32_with_input_A(backend, tensor_map["v.blk.0.ffn_up.weight"])) {
    
            all_passed = false;
        }        
        ggml_backend_buffer_free(buffer);
        gguf_free(gguf_ctx);
        ggml_free(ctx_gguf);



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
