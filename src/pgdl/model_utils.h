#pragma once
#ifndef _MODEL_UTILS_H_
#define _MODEL_UTILS_H_

#include "vector.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "fmgr.h"

#ifdef __cplusplus
}
#endif

typedef struct ModelLayer {
    char*    layer_name;
    MVec*    layer_parameter;
} ModelLayer;


bool get_mvec_oid(int32_t& oid);

bool get_model_layer_size(const char* model_name, int32_t& layer_size);

bool get_model_layer_name(const char* model_name, int32_t layer_index, std::string& layer_name);


bool compare_model_struct(const torch::jit::script::Module& model, const torch::jit::script::Module& base_model);

int compare_model_struct(const char* model_name, const char* base_model_name);

bool get_model_layer_parameter(const char* model_name, int32_t layer_index, torch::Tensor& tensor);

bool get_model_layer_parameter(const char* model_name, const char* layer_name, torch::Tensor& tensor);

bool insert_model_layer_parameter(const char* model_name, const char* layer_name, int32_t layer_index, int32_t oid, MVec* vector);

bool delete_model_parameter(const char* model_name);

void model_parameter_extraction(const char* model_path, const char* base_model_name, ModelLayer** parameter_list, int32_t& layer_size);

// void model_parameter_merging(const char* model_name, torch::jit::script::Module& model);

bool load_model_by_basemodel(const char* model_name, const char* base_model_name);


#endif