/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-04-23 23:46:27
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2024-08-17 22:55:14
 * @FilePath: /postgres-DB4AI/src/udf/torch_wrapper.h
 * @Description: 多模型管理类，包括可变参数的传递
 */
#pragma once
#ifndef _MODEL_MANAGER_H_
#define _MODEL_MANAGER_H_

#include "env.h"

#include <stdbool.h>
#include <string>
#include <unordered_map>

// callback parameter
typedef union {
    void* ptr;
    int   integer;
    float8 floating;
} Args;


class ModelManager{
public:
    using PreProcessCallback = bool(*)(std::vector<torch::jit::IValue>&, Args*);
    using OutputProcessFloatCallback = bool(*)(torch::jit::IValue&, Args*, float8&);
    using OutputProcessTextCallback = bool(*)(torch::jit::IValue&, Args*, std::string&);
    
    ModelManager();
    ~ModelManager();
    /**
     * @description: 
     * @event: 
     * @param {string } model_path
     * @return {*}
     */    
    bool LoadModel(std::string model_name, std::string  model_path);

    /**
     * @description: 
     * @event: 
     * @param {string} model_type 模型类别，如分类
     * @param {string} model_name 模型名
     * @param {string} model_path 模型文件在本地文件系统的路径
     * @param {string} discription 模型描述
     * @return {*}
     */    
    bool CreateModel(const std::string model_name, 
                     const std::string model_path,
                     const std::string base_model, 
                     const std::string discription);

    /**
     * @description: 
     * @event: 
     * @param {string} model_name
     * @param {string} model_path
     * @return {*}
     */    
    bool UpdateModel(const std::string model_name, 
                     const std::string model_path);

    /**
     * @description: 
     * @event: 
     * @param {string} model_name
     * @return {*}
     */    
    bool DropModel(const std::string model_name);

    /**
     * @description: 
     * @event: 
     * @param {string} model_name 
     * @param {string&} model_path
     * @return {*}
     */    
    bool GetModelPath(const std::string model_name, 
                      std::string& model_path);

    bool GetModelDeviceType(const std::string model_name,
                            torch::DeviceType& device_type);

    bool GetBaseModelPathFromBaseModel(const std::string base_model_name, 
                          std::string& base_model_path);

    bool GetBaseModelPathFromModel(const std::string model_name, 
                          std::string& base_model_path);

    bool HaveBaseModel(const std::string model_name);

    bool IsBaseModelExist(const std::string base_model_name);
    /**
     * @description: 
     * @event: 
     * @param {string} model_path
     * @param {string} model_name
     * @param {string&} md5 
     * @return {*}
     */    
    bool GetModelMd5(const std::string model_path, 
                     const std::string model_name,
                     std::string& md5);

    /**
     * @description: 
     * @return {*}
     */    
    bool GetLastModelVersion(const std::string model_name, 
                             int16& version);

    /**
     * @description: 
     * @event: 
     * @param {string&} model_path
     * @return {*}
     */    
    bool SetCuda(const std::string& model_path);

    /**
     * @description: 
     * @event: 
     * @param {string&} model_path
     * @return {*}
     */    
    bool IsCuda(const std::string& model_path);

    /**
     * @description: 输入预处理函数
     * @event: 
     * @param {string} model_path
     * @param {Tensor&} img_tensor 处理后的tensor
     * @param {Args*} args 可变参数
     * @return {*}
     */    
    bool PreProcess(const std::string model_path, 
                    std::vector<torch::jit::IValue>& img_tensor, 
                    Args* args);

    /**
     * @description: 输出处理函数，结果为float8，存放在result中
     * @event: 
     * @param {string} model_path
     * @param {Tensor&} output_tensor
     * @param {float8&} result 返回的处理的结果
     * @return {*}
     */    
    bool OutputProcessFloat(const std::string model_path, 
                            torch::jit::IValue& output_tensor, 
                            Args* args,
                            float8& result);

    /**
     * @description: 输出处理函数，结果为text，存放在result中
     * @event: 
     * @param {string} model_path
     * @param {Tensor&} output_tensor
     * @param {text*} result
     * @return {*}
     */    
    bool OutputProcessText(const std::string model_path, 
                           torch::jit::IValue& output_tensor, 
                           Args* args,
                           std::string& result);
    
    /**
     * @description: 注册输入处理回调函数
     * @event: 
     * @return {*}
     */
    void RegisterPreProcess(const std::string& model_name, 
                            PreProcessCallback func);

    /**
     * @description: 注册输出处理回调函数，返回值为float
     * @event: 
     * @param {string&} path
     * @param {OutputProcessFloatCallback} func
     * @return {*}
     */    
    void RegisterOutoutProcessFloat(const std::string& model_name, 
                                    OutputProcessFloatCallback func);

    /**
     * @description: 注册输出处理回调函数，返回值为text
     * @event: 
     * @param {string&} path
     * @param {OutputProcessTextCallback} func
     * @return {*}
     */    
    void RegisterOutoutProcessText(const std::string& model_name, 
                                   OutputProcessTextCallback func);

    bool Predict(const std::string& model_path,
                 std::vector<torch::jit::IValue>& input, 
                 torch::jit::IValue& output);
private:
    ModelManager(const ModelManager &other);  
    ModelManager & operator=(const ModelManager &other);
private:
    std::unordered_map<std::string, std::pair<torch::jit::script::Module, torch::DeviceType>>        module_handle_;  //key为路径，value为module句柄以及是否使用gpu
    std::unordered_map<std::string, PreProcessCallback>                                              module_preprocess_functions_; //key为模型路径，value为注册的预处理回调函数
    std::unordered_map<std::string, OutputProcessFloatCallback>                                      module_outputprocess_functions_float_; //key为模型路径，value为输出处理回调函数
    std::unordered_map<std::string, OutputProcessTextCallback>                                       module_outputprocess_functions_text_; //key为模型路径，value为输出处理回调函数
};


#endif