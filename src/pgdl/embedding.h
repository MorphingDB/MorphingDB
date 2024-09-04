#ifndef _MODEL_MD5_H_
#define _MODEL_MD5_H_


#include "vector.h"

MVec* 
image_to_vector(const int32_t width, 
                      const int32_t height, 
                      const float8 norm_mean,
                      const float8 norm_std,
                      const char* image_url);




MVec* 
text_to_vector(const char* text);




#endif