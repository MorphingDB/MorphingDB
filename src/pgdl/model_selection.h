/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2024-08-13 09:29:35
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2024-08-20 14:03:24
 * @FilePath: /pgdl_basemodel_new/src/pgdl/model_selection.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#ifndef _MODEL_SELECTION_H_
#define _MODEL_SELECTION_H_

#include "env.h"
#include <string>
#include <map>
#include <vector>


class ModelSelection{

public:
    ModelSelection(const std::string visual_model_path,
                   const std::string regression_model_path)
        : default_n_px(224),
          regression_output_dim(5),
          visual_model_path(visual_model_path),
          regression_model_path(regression_model_path)
    {
        
    }
    ~ModelSelection()
    {
        
    }

    std::string SelectModel(const std::string& table_name,
                             const std::string& col_name,
                             const int& sample_size,
                             std::string dataset = "mean");

private:
    torch::Tensor GetForwardClip(const std::vector<std::string>& data_list, 
                                   std::string visual_model_path);

    torch::Tensor Preprocess(const std::string& image_path, 
                             int n_px);

    std::vector<std::string> GetDataList(const std::string& table_name,
                                         const std::string& col_name,
                                         const int& sample_size);
    
private:
    std::vector<std::string> images_path;
    int default_n_px;
    int regression_output_dim;
    std::string visual_model_path;
    std::string regression_model_path;

};







#endif