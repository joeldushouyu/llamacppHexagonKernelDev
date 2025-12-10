#pragma once


#include <iostream>
#include <unordered_map>
#include <string>
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"


//NOTE: caller responsible to release backend buffer, gguf_ctx, ctx_gguf

bool  load_gguf_with_data(
    const std::string &filename, 
    ggml_backend_t backend, 

    // caller need to release this buffer
    struct gguf_context * gguf_ctx,
    struct ggml_context * ctx_gguf,
    ggml_backend_buffer_t &buffer,
    std::unordered_map<std::string, struct ggml_tensor*> &tensor_map
);

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
);