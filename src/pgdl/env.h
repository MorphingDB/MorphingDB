/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-05-07 14:10:23
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2023-07-06 17:22:23
 * @FilePath: /postgres-DB4AI/src/udf/env.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#ifndef _ENV_H_
#define _ENV_H_


#include "c10/core/DeviceType.h"
#include "c10/core/Device.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef LOG
#undef LOG
#endif

#include "postgres.h"
#include "fmgr.h"

#ifdef __cplusplus
}
#endif

//#include <opencv/cv.h>

// 为了使用 libintl.h 中的gettext dgettext等函数
//#define ENABLE_NLS




#endif
