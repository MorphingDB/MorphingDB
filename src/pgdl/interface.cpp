#include "model_manager.h"

#include "interface.h"
#include "unistd.h"
#include "myfunc.h"
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif
#include "utils/builtins.h"
#include "utils/formatting.h"
#include "catalog/pg_type_d.h"
#include "common/fe_memutils.h"



ModelManager      model_manager;

PG_MODULE_MAGIC;

// model manager functions
PG_FUNCTION_INFO_V1(create_model);
PG_FUNCTION_INFO_V1(modify_model);
PG_FUNCTION_INFO_V1(drop_model);

// prefict functions
PG_FUNCTION_INFO_V1(predict_float);
PG_FUNCTION_INFO_V1(predict_text);

// register external process
PG_FUNCTION_INFO_V1(register_process);

Datum
create_model(PG_FUNCTION_ARGS)
{
    char* model_name = NULL;
    char* model_path = NULL;
    char* discription = NULL;

    if(PG_NARGS() != 3){
        ereport(ERROR, (errmsg("CreateModel requires 3 parameters!")));
    }

    model_name = PG_GETARG_CSTRING(0);
    model_path = PG_GETARG_CSTRING(1);
    discription = PG_GETARG_CSTRING(2);

    if(strlen(model_name) == 0){
        ereport(ERROR, (errmsg("model_name is empty!")));
    }
    
    if(access(model_path, F_OK) != 0){
        ereport(ERROR, (errmsg("model is not exist!")));
    }

    if(model_manager.CreateModel(model_name, model_path, discription)){
        PG_RETURN_BOOL(true);
    }else {
        ereport(ERROR, (errmsg("create model error!")));
    }
    PG_RETURN_BOOL(false);
}

Datum
modify_model(PG_FUNCTION_ARGS)
{
    char* model_name = PG_GETARG_CSTRING(0);
    char* model_path = PG_GETARG_CSTRING(1);
	

    if(strlen(model_name) == 1){
        ereport(ERROR, (errmsg("model_name is empty!")));
    }
    
    if(access(model_path, F_OK) != 0){
        ereport(ERROR, (errmsg("model is not exist!")));
    }

    if(model_manager.UpdateModel(model_name, model_path)){
        PG_RETURN_BOOL(true);
    }else {
        ereport(ERROR, (errmsg("modify model error!")));
    }
    PG_RETURN_BOOL(false);
}

Datum
drop_model(PG_FUNCTION_ARGS)
{
    char* model_name = PG_GETARG_CSTRING(0);

    if(strlen(model_name) == 1){
        elog(ERROR,"model name is empty!");
        PG_RETURN_BOOL(false);
    }

    if(model_manager.DropModel(model_name)){
        PG_RETURN_BOOL(true);
    }else {
        elog(ERROR,"model not exist, don't need to drop!");
    }
    PG_RETURN_BOOL(false);
}


Datum
predict_float(PG_FUNCTION_ARGS)
{
    char*           model_name = NULL; 
    char*           cuda = NULL;
    char*           text_array = NULL;
    int             num_elems = 0;
    char*           ret_str = NULL;

    Args*           args = (Args*)palloc((PG_NARGS()-2) * sizeof(Args)); 

    std::string     model_path;
    Datum           datea;

    model_name = PG_GETARG_CSTRING(0);
    cuda       = PG_GETARG_CSTRING(1);

    // 1. load model
    if(strlen(model_name) == 0){
        ereport(ERROR, (errmsg("model name is empty!")));
    }

    if(!model_manager.GetModelPath(model_name, model_path)){
        ereport(ERROR, (errmsg("model not exist, can't get path!")));
    }
    
    if(!model_manager.LoadModel(model_path)){
        ereport(ERROR, (errmsg("load model error")));
    }

    // 2. choose cpu or gpu to predict
    if (pg_strcasecmp(cuda, "gpu") && 
        model_manager.SetCuda(model_path)){
        
    }

    // 3. get all col for input
    for (int i = 2; i < PG_NARGS(); i++) {
        switch (get_fn_expr_argtype(fcinfo->flinfo, i)) {
            case INT4OID:
            case INT2OID:
            case INT8OID:
            {
                int cur_int = PG_GETARG_INT32(i);
                args[i-2].integer = cur_int;
                break;
            }
            case FLOAT4OID:
            case FLOAT8OID:
            {
                float8 cur_float = PG_GETARG_FLOAT8(i);
                args[i-2].floating = cur_float;
                break;
            }
            case TEXTOID:
            {
                char* cur_text = TextDatumGetCString(PG_GETARG_DATUM(i));
                args[i-2].ptr = cur_text;
                break;
            }
            case CSTRINGOID:
            {
                char* cur_cstring = PG_GETARG_CSTRING(i);
                args[i-2].ptr = cur_cstring;
                break;
            }
            case NUMERICOID:
            {
                Datum numer = PG_GETARG_DATUM(i);
                float8 num_float = DatumGetFloat8(DirectFunctionCall1(numeric_float8, numer));;
                args[i-2].floating = num_float;
                break;
            }
            default:
            {
                ereport(ERROR, (errmsg("%d type don't support!", get_fn_expr_argtype(fcinfo->flinfo, i))));
                break;
            }
        }
    }

    // 4. run preprocess callback function
    std::vector<torch::jit::IValue> preprecess_tensor;
    torch::jit::IValue output_tensor;
    if(!model_manager.PreProcess(model_path, preprecess_tensor, args)){
        ereport(ERROR, (errmsg("preprocess error!")));
    }

    // 5. predict by preprocess output
    if(!model_manager.Predict(model_path, preprecess_tensor, output_tensor)){
        ereport(ERROR, (errmsg("predict error!")));
    }

    // 6. run outputprocess callback funtion
    float8 result;
    if(!model_manager.OutputProcessFloat(model_path, output_tensor, args, result)){
        ereport(ERROR, (errmsg("output process error!")));
    }
    

    Datum ret = Float8GetDatum(result);
    PG_RETURN_DATUM(ret);
}

Datum
predict_text(PG_FUNCTION_ARGS)
{
    char*           model_name = NULL; 
    char*           cuda = NULL;
    char*           text_array = NULL;
    int             num_elems = 0;
    char*           ret_str = NULL;

    Args*           args = (Args*)palloc((PG_NARGS()-2) * sizeof(Args)); 

    std::string     model_path;
    Datum           datea;

    model_name = PG_GETARG_CSTRING(0);
    cuda       = PG_GETARG_CSTRING(1);

     // 1. load model
    if(strlen(model_name) == 0){
        ereport(ERROR, (errmsg("model name is empty!")));
    }

    if(!model_manager.GetModelPath(model_name, model_path)){
        ereport(ERROR, (errmsg("model not exist, can't get path!")));
    }
    
    if(!model_manager.LoadModel(model_path)){
        ereport(ERROR, (errmsg("load model error")));
    }

    // 2. choose cpu or gpu to predict
    if (pg_strcasecmp(cuda, "gpu") && 
        model_manager.SetCuda(model_path)){
        
    }

    // 3. get all col for input
    for (int i = 2; i < PG_NARGS(); i++) {
        switch (get_fn_expr_argtype(fcinfo->flinfo, i)) {
            case INT4OID:
            case INT2OID:
            case INT8OID:
            {
                int cur_int = PG_GETARG_INT32(i);
                args[i-2].integer = cur_int;
                break;
            }
            case FLOAT4OID:
            case FLOAT8OID:
            {
                float8 cur_float = PG_GETARG_FLOAT8(i);
                args[i-2].floating = cur_float;
                break;
            }
            case TEXTOID:
            {
                char* cur_text = TextDatumGetCString(PG_GETARG_DATUM(i));
                args[i-2].ptr = cur_text;
                break;
            }
            case CSTRINGOID:
            {
                char* cur_cstring = PG_GETARG_CSTRING(i);
                args[i-2].ptr = cur_cstring;
                break;
            }
            case NUMERICOID:
            {
                Datum numer = PG_GETARG_DATUM(i);
                float8 num_float = DatumGetFloat8(DirectFunctionCall1(numeric_float8, numer));;
                args[i-2].floating = num_float;
                break;
            }
            default:
            {
                ereport(ERROR, (errmsg("%d type don't support!", get_fn_expr_argtype(fcinfo->flinfo, i))));
                break;
            }
        }
    }

    // 4. run preprocess callback function
    // 4. run preprocess callback function
    std::vector<torch::jit::IValue> preprecess_tensor;
    torch::jit::IValue output_tensor;
    if(!model_manager.PreProcess(model_path, preprecess_tensor, args)){
        ereport(ERROR, (errmsg("preprocess error!")));
    }

    // 5. predict by preprocess output
    if(!model_manager.Predict(model_path, preprecess_tensor, output_tensor)){
        ereport(ERROR, (errmsg("predict error!")));
    }
    
    // 6. run outputprocess callback funtion
    text* result = nullptr;
    std::string result_str;
    if(!model_manager.OutputProcessText(model_path, output_tensor, args, result_str)){

    }

    result = (text*)palloc(result_str.size() + VARHDRSZ);
    SET_VARSIZE(result, result_str.size() + VARHDRSZ);
    memcpy(VARDATA(result), result_str.c_str(), result_str.size());

    pfree(args);
    PG_RETURN_TEXT_P(PointerGetDatum(result));
}

Datum
register_process(PG_FUNCTION_ARGS)
{
    register_callback();
    PG_RETURN_VOID();
}

#ifdef __cplusplus
}
#endif
