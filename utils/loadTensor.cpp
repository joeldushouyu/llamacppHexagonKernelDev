#include "loadTensor.hpp"


bool load_tensor_from_bin_file(
    const std::string &filename,
    ggml_backend_t backend,
    ggml_backend_buffer_t buffer,
    const std::string &tensor_name,
    const int ne[GGML_MAX_DIMS],
    const int nb[GGML_MAX_DIMS],
    enum ggml_type type,
    //Note: caller responsible to release this buffer
    struct ggml_tensor* tensor
){
    // Open binary file
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        GGML_LOG_ERROR("Failed to open binary file: %s\n", filename.c_str());
        return false;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Calculate expected tensor size based on dimensions
    size_t expected_size = ggml_nbytes(tensor);
    
    // Verify file size matches expected tensor size
    if (file_size != expected_size) {
        GGML_LOG_ERROR("File size mismatch for %s: file=%zu bytes, expected=%zu bytes\n", 
                       filename.c_str(), file_size, expected_size);
        GGML_LOG_ERROR("Tensor dimensions: [%d, %d, %d, %d], type=%s\n",
                       ne[0], ne[1], ne[2], ne[3], ggml_type_name(type));
        fclose(file);
        return false;
    }
    
    // Verify buffer is valid
    if (!buffer) {
        GGML_LOG_ERROR("Invalid buffer provided for tensor %s\n", tensor_name.c_str());
        fclose(file);
        return false;
    }
    
    // Only allocate if tensor doesn't already have a buffer assigned
    if (tensor->buffer == NULL) {
        ggml_backend_tensor_alloc(buffer, tensor, NULL);
    }
    
    // Read data into temporary CPU buffer
    void* temp_data = malloc(file_size);
    if (!temp_data) {
        GGML_LOG_ERROR("Failed to allocate temporary buffer of size %zu\n", file_size);
        fclose(file);
        return false;
    }
    
    size_t bytes_read = fread(temp_data, 1, file_size, file);
    fclose(file);
    
    if (bytes_read != file_size) {
        GGML_LOG_ERROR("Failed to read complete file: read %zu of %zu bytes\n", 
                       bytes_read, file_size);
        free(temp_data);
        return false;
    }
    
    // Transfer data from CPU buffer to backend tensor
    ggml_backend_tensor_set(tensor, temp_data, 0, file_size);
    
    free(temp_data);
    
    GGML_LOG_INFO("Successfully loaded tensor '%s' from %s (%zu bytes) to backend\n", 
                  tensor_name.c_str(), filename.c_str(), file_size);
    
    return true;
}

bool  load_gguf_with_data(
    const std::string &filename, 
    ggml_backend_t backend, 

    // caller need to release this buffer
    struct gguf_context * gguf_ctx,
    struct ggml_context * ctx_gguf,
    ggml_backend_buffer_t &buffer,
    std::unordered_map<std::string, struct ggml_tensor*> &tensor_map
){
    

    GGML_LOG_INFO("\n=== Loading GGUF File with Data: %s ===\n", filename.c_str());

    // Step 1: Initialize GGUF context without allocating data
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ctx_gguf
    };
    gguf_ctx = gguf_init_from_file(filename.c_str(), params);
    
    if (!gguf_ctx) {
        GGML_LOG_ERROR("Failed to load GGUF file: %s\n", filename.c_str());
        return false;
    }
    
    // Step 2: Allocate tensors on backend (Hexagon)
    buffer = ggml_backend_alloc_ctx_tensors(ctx_gguf, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer for tensors\n");
        gguf_free(gguf_ctx);
        ggml_free(ctx_gguf);
        return false;
    }
    
    // Step 3: Load tensor data from file
    // You need to read the actual tensor data from the file
    // This typically involves seeking to the data section and reading raw bytes
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        GGML_LOG_ERROR("Failed to open file for reading tensor data\n");
        ggml_backend_buffer_free(buffer);
        gguf_free(gguf_ctx);
        ggml_free(ctx_gguf);
        return false;
    }
    
    // Get data offset (where tensor data starts in the file)
    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    fseek(file, data_offset, SEEK_SET);
    
    // Read each tensor's data and populate the tensor map
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    GGML_LOG_INFO("Loading %d tensors...\n", n_tensors);
    
    for (int i = 0; i < n_tensors; i++) {
        const char* tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_gguf, tensor_name);
        
        if (tensor) {
            size_t tensor_size = ggml_nbytes(tensor);
            
            // Read data into temporary CPU buffer
            void* temp_data = malloc(tensor_size);
            size_t read_size = fread(temp_data, 1, tensor_size, file);
            
            if (read_size != tensor_size) {
                GGML_LOG_ERROR("Failed to read tensor data for %s\n", tensor_name);
                free(temp_data);
                continue;
            }
            
            // Copy data to backend buffer
            ggml_backend_tensor_set(tensor, temp_data, 0, tensor_size);
            
            // Add tensor to the map
            tensor_map[std::string(tensor_name)] = tensor;
            
            GGML_LOG_INFO("Loaded tensor [%d/%d]: %s (%zu bytes)\n", i+1, n_tensors, tensor_name, tensor_size);
            free(temp_data);
        }
    }
    
    fclose(file);
    
    GGML_LOG_INFO("\n=== All %d tensors loaded successfully ===\n", n_tensors);
    GGML_LOG_INFO("Tensor map size: %zu\n", tensor_map.size());
    
    // Now you can use the tensors via the tensor_map for inference
    // Example: auto my_tensor = tensor_map["token_embd.weight"];
    
    // Don't forget to cleanup when done:
    // ggml_backend_buffer_free(buffer);
    // gguf_free(gguf_ctx);
    // ggml_free(ctx_gguf);
    

    return true;

}