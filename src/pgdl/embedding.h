/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2024-08-08 11:25:33
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2024-09-04 15:36:09
 * @FilePath: /pgdl_basemodel_new/src/pgdl/embedding.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef _MODEL_MD5_H_
#define _MODEL_MD5_H_


#include "vector.h"

MVec* 
image_to_vector(const int32_t width, 
                      const int32_t height, 
                      const float8 norm_mean_1,
                      const float8 norm_mean_2,
                      const float8 norm_mean_3,
                      const float8 norm_std_1,
                      const float8 norm_std_2,
                      const float8 norm_std_3,
                      const char* image_url);




MVec* 
text_to_vector(const char* piece_model_path, 
               const char* text);





#endif