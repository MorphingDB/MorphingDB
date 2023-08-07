
#pragma once
#ifndef _INTERFACE_H_
#define _INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "fmgr.h"

/**
 * @description: accumulate one vector
 * @param: {internal}   aggregate state
 * @param: {cstring}    model name
 * @param: {cstring}    model path
 * @param: {any...}     vector
 * @return {internal} 
 */
Datum predict_batch_accum(PG_FUNCTION_ARGS);

/**
 * @description: discard batch vectors when batch are used
 * @param: {internal}   aggregate state
 * @param: {cstring}    model name
 * @param: {cstring}    model path
 * @param: {any...}     vector
 * @return {internal}  
 */
Datum predict_batch_accum_inv(PG_FUNCTION_ARGS);

/**
 * @description: fetch one result from batch
 * @param: {internal}   aggregate state
 * @return {float8}   
 */
Datum predict_batch_final_float(PG_FUNCTION_ARGS);

/**
 * @description: fetch one result from batch
 * @param: {internal}   aggregate state
 * @return {text}   
 */
Datum predict_batch_final_text(PG_FUNCTION_ARGS);

/**
 * @description: print time cost in pre, infer, post stage
 * @param: {bool}   enable printing
 * @return {bool}   0 as success   
 */
Datum enable_print_batch_time(PG_FUNCTION_ARGS);

/**
 * @description: predict_batch_xxx should only be called in window, this prevent call predict_batch_xxx as agg function
 * @param: {any...}   aggregate state
 * @return {bool} 0 as success, but this will never return
 */
Datum predict_batch_dummy(PG_FUNCTION_ARGS);

#ifdef __cplusplus
}
#endif

#endif