#include "model_manager.h"

#include "interface.h"
#include "unistd.h"
#include "myfunc.h"
#include "vector.h"
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif
#include "utils/builtins.h"
#include "utils/formatting.h"
#include "catalog/pg_type_d.h"
#include "catalog/namespace.h"
#include "common/fe_memutils.h"
#include "access/tableam.h"
#include "libpq/pqformat.h"
#include "utils/syscache.h"





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

// vector
PG_FUNCTION_INFO_V1(mvec_input);
PG_FUNCTION_INFO_V1(mvec_output);
PG_FUNCTION_INFO_V1(mvec_receive);
PG_FUNCTION_INFO_V1(mvec_send);

PG_FUNCTION_INFO_V1(get_mvec_data);
PG_FUNCTION_INFO_V1(get_mvec_shape);

// type conversion function
PG_FUNCTION_INFO_V1(array_to_mvec);
PG_FUNCTION_INFO_V1(text_to_mvec);
PG_FUNCTION_INFO_V1(mvec_to_float_array);

// operator function
PG_FUNCTION_INFO_V1(mvec_add);
PG_FUNCTION_INFO_V1(mvec_sub);
PG_FUNCTION_INFO_V1(mvec_equal);

PG_FUNCTION_INFO_V1(mvec_am_handler);

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

Datum
mvec_input(PG_FUNCTION_ARGS)
{
    char            *str = NULL;
    float           *x = (float*)palloc(sizeof(float) * MAX_VECTOR_DIM);
    int32           shape[MAX_VECTOR_SHAPE_SIZE];
    MVec            *vector = NULL;
    unsigned int    dim = 0;
    unsigned int    shape_size = 0;
    unsigned int    shape_dim = 1;

    
    str = PG_GETARG_CSTRING(0);

    ereport(INFO,
					(errmsg("str:%s",str)));

    parse_vector_str(str, &dim, x, &shape_size, shape);

    // Verify that the multiplication of shape values equals dim
    for(int i=0; i<shape_size; i++){
        shape_dim *= shape[i];
    }

    if(dim != shape_dim){
        ereport(ERROR,
					(errmsg("the multiplication of shape values not equals, dim:%d, shape_dim:%d", dim, shape_dim)));
    }
    vector = new_mvec(dim, shape_size);

    for(int i=0; i<dim; ++i){
       SET_MVEC_VAL(vector, i, x[i]);
    }

    for(int i=0; i<shape_size; ++i){
        ereport(INFO,
					(errmsg("shape[i]:%d", shape[i])));
        SET_MVEC_SHAPE_VAL(vector, i, shape[i]);

        ereport(INFO,
					(errmsg("shape1[i]:%d", GET_MVEC_SHAPE_VAL(vector, 0))));
    }

    pfree(x);
    PG_RETURN_POINTER(vector);
}

Datum
mvec_output(PG_FUNCTION_ARGS)
{
    MVec*           mvec = NULL;
    StringInfoData  ret;
    int32           dim = 0;
    int32           shape_size = 0;
    int             i = 0;

    mvec = PG_GETARG_MVEC_P(0);

    dim = GET_MVEC_DIM(mvec);
    shape_size = GET_MVEC_SHAPE_SIZE(mvec);

    if(dim > MAX_VECTOR_DIM){
        ereport(ERROR,
                (errmsg("dim is larger than %d dim!", MAX_VECTOR_DIM)));
    }

    if(shape_size > MAX_VECTOR_SHAPE_SIZE){
        ereport(ERROR,
					(errmsg("shape size is larger than 10!")));
    }

    initStringInfo(&ret);
    appendStringInfoChar(&ret, '[');

    if(dim > 10){
        for(int i=0; i<3; ++i){
            appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(float4out, Float4GetDatum(GET_MVEC_VAL(mvec, i)))));
            appendStringInfoChar(&ret, ',');
        }
        appendStringInfoString(&ret, "....");
        appendStringInfoChar(&ret, ',');
        for(int i=dim-3; i<dim; ++i){
            appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(float4out, Float4GetDatum(GET_MVEC_VAL(mvec, i)))));
            if (i != dim - 1) {
                appendStringInfoChar(&ret, ',');
            }
        }
    }else{
        for(int i=0; i<dim; ++i){
            appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(float4out, Float4GetDatum(GET_MVEC_VAL(mvec, i)))));
            if (i != dim - 1) {
                appendStringInfoChar(&ret, ',');
            }
        }
    }

    appendStringInfoChar(&ret, ']');
    
    appendStringInfoChar(&ret, '{');
    for(int i=0; i<shape_size; ++i){
        appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(int4out, Int32GetDatum(GET_MVEC_SHAPE_VAL(mvec, i)))));
        if (i != shape_size - 1) {
            appendStringInfoChar(&ret, ',');
        }
    }
    appendStringInfoChar(&ret, '}');

    PG_RETURN_CSTRING(ret.data);
}

Datum
mvec_receive(PG_FUNCTION_ARGS)
{
    StringInfo        str;
    MVec*             ret = NULL;
    int32_t           dim = 0;
    int32_t           shape_size = 0;
    int               i;

    str = (StringInfo)PG_GETARG_POINTER(0);
    dim = pq_getmsgint(str, sizeof(int32_t));

    ret = new_mvec(dim, 1);
    for(i=0; i<dim; ++i){
        SET_MVEC_VAL(ret, i, pq_getmsgfloat4(str));
    }

    PG_RETURN_POINTER(ret);
}

Datum
mvec_send(PG_FUNCTION_ARGS)
{
    MVec            *mvec; 
	StringInfoData   str;
    int              i;

    mvec = PG_GETARG_MVEC_P(0);

	pq_begintypsend(&str);
	pq_sendint(&str, GET_MVEC_DIM(mvec), sizeof(int32_t));
	for (i = 0; i < GET_MVEC_DIM(mvec); ++i){
        pq_sendfloat4(&str, GET_MVEC_VAL(mvec, i));
    }

	PG_RETURN_BYTEA_P(pq_endtypsend(&str));
}

Datum
get_mvec_data(PG_FUNCTION_ARGS)
{
    MVec	        *vector = NULL;
    //float           x[MAX_VECTOR_DIM];
    ArrayType       *result = NULL;
    Datum           *elems = NULL; 
    unsigned int    dim = 0;
    
    vector = PG_GETARG_MVEC_P(0);

    dim = GET_MVEC_DIM(vector);

    elems = (Datum *)palloc(sizeof(Datum *) * dim);

    for(int i=0; i<dim; ++i){
        elems[i] = Float4GetDatum(GET_MVEC_VAL(vector, i));
    }

    result = construct_array(elems, dim, FLOAT4OID, sizeof(float4), true, 'i');

    pfree(elems);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum
get_mvec_shape(PG_FUNCTION_ARGS)
{
    MVec	        *vector = NULL;
    //float           x[MAX_VECTOR_DIM];
    ArrayType       *result = NULL;
    Datum           *elems = NULL; 
    unsigned int    shape_size = 0;
    
    vector = PG_GETARG_MVEC_P(0);

    shape_size = GET_MVEC_SHAPE_SIZE(vector);

    elems = (Datum *)palloc(sizeof(Datum *) * shape_size);

    for(int i=0; i<shape_size; ++i){
        elems[i] = Int32GetDatum(GET_MVEC_SHAPE_VAL(vector, i));
    }

    result = construct_array(elems, shape_size, INT4OID, sizeof(int32), true, 'i');

    pfree(elems);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum
array_to_mvec(PG_FUNCTION_ARGS)
{
    ArrayType       *array = NULL;
    Oid             array_type;
    MVec            *mvec = NULL;
    int             array_length;
    int             i = 0;
    Datum           *elems = NULL;
    bool            *nulls = NULL;


    array = PG_GETARG_ARRAYTYPE_P(0);

    array_type = ARR_ELEMTYPE(array);

    
    switch (array_type) {
        case FLOAT4OID:
        {
            deconstruct_array(array, FLOAT4OID, sizeof(float4), true, 'i', &elems, &nulls, &array_length);
            break;
        }
        case FLOAT8OID:
        {
            deconstruct_array(array, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd', &elems, &nulls, &array_length);
            for(i=0; i<array_length; ++i){
                elems[i] = Float4GetDatum(DatumGetFloat8(elems[i]));
            }
            break;
        }
        case INT4OID:
        {
            deconstruct_array(array, INT4OID, sizeof(int), true, 'i', &elems, &nulls, &array_length);
            for(i=0; i<array_length; ++i){
                elems[i] = Float4GetDatum(DatumGetInt32(elems[i]));
            }
            break;
        }
        default:
        {
            ereport(ERROR,
					(errmsg("unsupport %d type to mvec!"), array_type));
        }
    }

    if(array_length > MAX_VECTOR_DIM){
        ereport(ERROR,
					(errmsg("mvec cannot have more than %d dimensions", MAX_VECTOR_DIM)));
    }

    mvec = new_mvec(array_length, 1);

    for(i=0; i<array_length; ++i){
        SET_MVEC_VAL(mvec, i, DatumGetFloat4(elems[i]));
    }

    SET_MVEC_SHAPE_VAL(mvec, 0, array_length);

    PG_RETURN_POINTER(mvec);

}

Datum
text_to_mvec(PG_FUNCTION_ARGS)
{
    char             *str = NULL;
    MVec             *vector = NULL;
    unsigned int     dim = 0;

    int32            shape[MAX_VECTOR_SHAPE_SIZE];
    float            *x = (float*)palloc(sizeof(float) * MAX_VECTOR_DIM);
    unsigned int     shape_size = 0;
    unsigned int     shape_dim = 1;

    str = TextDatumGetCString(PG_GETARG_DATUM(0));

    parse_vector_str(str, &dim, x, &shape_size, shape);

    for(int i=0; i<shape_size; i++){
        shape_dim *= shape[i];
    }

    if(dim != shape_dim){
        ereport(ERROR,
					(errmsg("the multiplication of shape values not equals, dim:%d, shape_dim:%d", dim, shape_dim)));
    }

    vector = new_mvec(dim, shape_size);
    
    for(int i=0; i<dim; ++i){
        SET_MVEC_VAL(vector, i, x[i]);
    }

    for(int i=0; i<shape_size; ++i){
        SET_MVEC_SHAPE_VAL(vector, i, shape[i]);
    }

    pfree(x);
    PG_RETURN_POINTER(vector);
}

Datum
mvec_to_float_array(PG_FUNCTION_ARGS)
{
    MVec	        *mvec = NULL;
    ArrayType       *ret = NULL;
    Datum           *elems = NULL; 
    int32_t         dim = 0;
    int             i = 0;
    
    mvec = PG_GETARG_MVEC_P(0);

    dim = GET_MVEC_DIM(mvec);

    elems = (Datum *)palloc(sizeof(Datum *) * dim);

    for(i=0; i<dim; ++i){
        elems[i] = Float4GetDatum(GET_MVEC_VAL(mvec, i));
    }

    ret = construct_array(elems, dim, FLOAT4OID, sizeof(float4), true, 'i');

    pfree(elems);
    PG_RETURN_ARRAYTYPE_P(ret);
}

Datum
mvec_add(PG_FUNCTION_ARGS)
{
    MVec	      *mvec_left = NULL;
    MVec          *mvec_right = NULL;
    MVec          *ret = NULL;
    int32_t        dim = 0;
    int32_t        shape_size = 0;
    int            i = 0;

    mvec_left = PG_GETARG_MVEC_P(0);
    mvec_right = PG_GETARG_MVEC_P(1);


    if(GET_MVEC_DIM(mvec_left) != GET_MVEC_DIM(mvec_right)){
        ereport(ERROR,
					(errmsg("the two mvecs have different dimensions!,(),()")));
    }

    if(!shape_equal(mvec_left, mvec_right)){
        ereport(ERROR,
					(errmsg("the two mvecs have different shape!,(),()")));
    }
    
    dim = GET_MVEC_DIM(mvec_left);
    shape_size = GET_MVEC_SHAPE_SIZE(mvec_left);
    ret = new_mvec(dim, shape_size);
    
    for(i=0; i<dim; ++i){
        float left = GET_MVEC_VAL(mvec_left, i);
        float right = GET_MVEC_VAL(mvec_right, i);
        float res = left + right;
        if (unlikely(isinf(res)) && !isinf(left) && !isinf(right)){
            ereport(ERROR,
					(errmsg("overflow for %f + %f", left, right)));
        }
        SET_MVEC_VAL(ret, i, (left + right));
    }

    for(i=0; i<shape_size; ++i){
        int32_t value = GET_MVEC_SHAPE_VAL(mvec_left, i);
        SET_MVEC_SHAPE_VAL(ret, i, value);
    }

    PG_RETURN_POINTER(ret);
}

Datum
mvec_sub(PG_FUNCTION_ARGS)
{
    MVec	      *mvec_left = NULL;
    MVec          *mvec_right = NULL;
    MVec          *ret = NULL;
    int32_t        dim = 0;
    int32_t        shape_size = 0;
    int            i = 0;

    mvec_left = PG_GETARG_MVEC_P(0);
    mvec_right = PG_GETARG_MVEC_P(1);


    if(GET_MVEC_DIM(mvec_left) != GET_MVEC_DIM(mvec_right)){
        ereport(ERROR,
					(errmsg("the two mvecs have different dimensions!,(),()")));
    }

    if(!shape_equal(mvec_left, mvec_right)){
        ereport(ERROR,
					(errmsg("the two mvecs have different shape!,(),()")));
    }

    dim = GET_MVEC_DIM(mvec_left);
    shape_size = GET_MVEC_SHAPE_SIZE(mvec_left);
    ret = new_mvec(dim, shape_size);
    
    for(i=0; i<dim; ++i){
        float left = GET_MVEC_VAL(mvec_left, i);
        float right = GET_MVEC_VAL(mvec_right, i);
        float res = left - right;
        if (unlikely(isinf(res)) && !isinf(left) && !isinf(right)){
            ereport(ERROR,
					(errmsg("overflow for %f - %f", left, right)));
        }
        SET_MVEC_VAL(ret, i, res);
    }

    for(i=0; i<shape_size; ++i){
        int32_t value = GET_MVEC_SHAPE_VAL(mvec_left, i);
        SET_MVEC_SHAPE_VAL(ret, i, value);
    }


    PG_RETURN_POINTER(ret);
}

Datum
mvec_equal(PG_FUNCTION_ARGS)
{
    MVec	      *mvec_left = NULL;
    MVec          *mvec_right = NULL;
    MVec          *ret = NULL;
    int32_t        dim = 0;
    int            i = 0;

    mvec_left = PG_GETARG_MVEC_P(0);
    mvec_right = PG_GETARG_MVEC_P(1);


    if(GET_MVEC_DIM(mvec_left) != GET_MVEC_DIM(mvec_right)){
        PG_RETURN_BOOL(false);
    }

    dim = GET_MVEC_DIM(mvec_left);
    
    for(i=0; i<dim; ++i){
        float left = GET_MVEC_VAL(mvec_left, i);
        float right = GET_MVEC_VAL(mvec_right, i);
        if (isnan(left) ? !isnan(right) 
                : (isnan(right) || left - right > 1e-6 || right - left > 1e-6))
            PG_RETURN_BOOL(false);
    }

    PG_RETURN_BOOL(true);
}

#ifdef __cplusplus
}
#endif
