/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-04-24 17:09:43
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2023-07-25 11:50:35
 * @FilePath: /postgres-DB4AI/src/udf/interface.h
 * @Description: udf对外提供的接口函数
 */
#pragma once
#ifndef _INTERFACE_H_
#define _INTERFACE_H_


#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "fmgr.h"
/**
 * @description: 
 * @param: {cstring}    model name
 * @param: {cstring}    model path
 * @return {bool}
 */
Datum create_model(PG_FUNCTION_ARGS);
/**
 * @description: 
 * @param: {cstring}    model name
 * @param: {cstring}    model path
 * @return {bool}
 */
Datum modify_model(PG_FUNCTION_ARGS);
/**
 * @description: 
 * @param: {cstring}    model name
 * @return {bool}
 */
Datum drop_model(PG_FUNCTION_ARGS);


/**
 * @description: 
 * @event: 
 * @return float8
 */
Datum predict_float(PG_FUNCTION_ARGS);

/**
 * @description: 
 * @event: 
 * @return text
 */
Datum prefict_text(PG_FUNCTION_ARGS);

/**
 * @description: 
 * @return {*}
 */
Datum register_process(PG_FUNCTION_ARGS);

#ifdef __cplusplus
}
#endif

#endif