#include <thread>
#include <vector>
#include "ATen/core/TensorBody.h"


#include "model_manager.h"
#include "model_utils.h"
#include "vector.h"

extern "C" {
#include "postgres.h"

#include "catalog/pg_type_d.h"
#include "executor/nodeAgg.h"
#include "fmgr.h"
#include "nodes/pg_list.h"
#include "utils/builtins.h"
#include "utils/palloc.h"

typedef struct TimeFilter {
    int64_t load_model_time; //ms
    int64_t pre_time;   // ms
    int64_t infer_time; // ms
    int64_t post_time;  // ms
} TimeFilter;

// args are "state, model, cuda, vector_elements.."
#define VECTOR_START_ARG_INDEX 3

static bool debug_print_batch_time;
extern ModelManager model_manager;
extern TimeFilter time_filter;

typedef struct VecAggState {
    MemoryContext ctx;
    List* ins;
    List* outs;
    int batch_i;
    int prcsd_batch_n;
    char* model;
    char* cuda;
    int nxt_csr;
    int64_t pre_time;   // ms
    int64_t infer_time; // ms
    int64_t post_time;  // ms
} VecAggState;

// batch prefict functions
PG_FUNCTION_INFO_V1(predict_batch_accum);
PG_FUNCTION_INFO_V1(predict_batch_accum_inv);
PG_FUNCTION_INFO_V1(predict_batch_final_float);
PG_FUNCTION_INFO_V1(predict_batch_final_text);
PG_FUNCTION_INFO_V1(enable_print_batch_time);
PG_FUNCTION_INFO_V1(predict_batch_dummy);

VecAggState *
makeVecAggState(FunctionCallInfo fcinfo)
{
    VecAggState *state;
    MemoryContext agg_context;
    MemoryContext old_context;

    if (!AggCheckCallContext(fcinfo, &agg_context))
        elog(ERROR, "aggregate function called in non-aggregate context");

    old_context = MemoryContextSwitchTo(agg_context);

    state = (VecAggState *) palloc0(sizeof(VecAggState));
    state->ctx = agg_context;
    state->model = pstrdup(PG_GETARG_CSTRING(1));
    state->cuda = pstrdup(PG_GETARG_CSTRING(2));

    MemoryContextSwitchTo(old_context);

    return state;
}

Args* 
makeVecFromArgs(FunctionCallInfo fcinfo, int start, int dim) 
{
    int mvec_oid = 0;

    Args* vec = (Args*) palloc0(sizeof(Args) * dim);
    if(!get_mvec_oid(mvec_oid)){
        ereport(ERROR, (errmsg("get mvec oid error!")));
    }

    for (int i = start; i < dim + start; i++) {
        Oid argtype = get_fn_expr_argtype(fcinfo->flinfo, i);
        if(argtype == INT4OID || argtype == INT2OID || argtype == INT8OID){
            int cur_int = PG_GETARG_INT32(i);
            vec[i-start].integer = cur_int;
        }else if(argtype == FLOAT4OID || argtype == FLOAT8OID){
            float8 cur_float = PG_GETARG_FLOAT8(i);
            vec[i-start].floating = cur_float;
        }else if(argtype == TEXTOID){
            char* cur_text = TextDatumGetCString(PG_GETARG_DATUM(i));
            vec[i-start].ptr = cur_text;
        }else if(argtype == CSTRINGOID){
            char* cur_cstring = PG_GETARG_CSTRING(i);
            vec[i-start].ptr = cur_cstring;
        }else if(argtype == NUMERICOID){
            Datum numer = PG_GETARG_DATUM(i);
            float8 num_float = DatumGetFloat8(DirectFunctionCall1(numeric_float8, numer));;
            vec[i-start].floating = num_float;
        }else if(argtype == mvec_oid){
            MVec* cur_mvec = DatumGetMVec(PG_GETARG_DATUM(i));
            vec[i-start].ptr = cur_mvec;
        }else{
            ereport(ERROR, (errmsg("%d type don't support!", get_fn_expr_argtype(fcinfo->flinfo, i))));
        }
    }
    return vec;
}

static bool
wait_and_check_error(std::vector<std::thread> &pool, std::vector<int> &res, int parallel_num)
{
    bool has_error = false;
    for (auto &t : pool)
            t.join();

    for (int i = 0; i < parallel_num; i++)
    {
            if (!res[i])
            {
                has_error = true;
                break;
            }
    }

    pool.clear();
    res.clear();
    return has_error;
}

#define CLEAN_UP_CPP_OBJS() \
input_tensors.clear();      \
batch_inputs_tmp.clear();   \
input_batch_tensor.clear(); \
output.~IValue();    \
outputs.clear()

#define WAIT_AND_CHECK_ERROR(stage)                            \
if (wait_and_check_error(pool, res, prcsd_batch_n))            \
{                                                              \
    CLEAN_UP_CPP_OBJS();                                        \
    ereport(ERROR, (errmsg("meet error in " stage " stage"))); \
}

#define CLOCK_START() auto start = std::chrono::system_clock::now()
#define CLOCK_END(type) time_filter.type##_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count()

static std::vector<torch::jit::IValue>
split_results_one_level(torch::jit::IValue res) {
    std::vector<torch::jit::IValue> rets;
    if (res.isTensor()) {
        auto&& tensors = torch::split(res.toTensor(), 1, 0);
        for (auto&& t: tensors) {
            rets.emplace_back(torch::jit::IValue(std::move(t)));
        }
    } else if (res.isList()) {
        auto&& list = res.toList();
        for (auto&& it: list) {
            rets.emplace_back(std::move(it));
        }
    } else if (res.isTuple()) {
        auto&& eles = res.toTuple()->elements();
        for (auto&& e : eles) {
            rets.emplace_back(torch::jit::IValue(std::move(c10::ivalue::Tuple::create(e))));
        }
    }
    return rets;
}

static std::vector<torch::jit::IValue>
split_results(torch::jit::IValue output) {
    std::vector<torch::jit::IValue> ret;
    if (output.isTensor()) {
        return split_results_one_level(output);
    } else if (output.isTuple()) {
        std::vector<std::vector<torch::jit::IValue>> cols;
        for (auto&& it : output.toTuple()->elements()) {
            cols.emplace_back(split_results_one_level(std::move(it)));
        }
        // col to row
        int row_num = cols[0].size();
        std::vector<torch::jit::IValue> row;
        for (int i = 0; i < row_num; i++) {
            for (int j = 0; j < cols.size(); j++) {
                row.emplace_back(std::move(cols[j][i]));
            }
            ret.emplace_back(torch::jit::IValue(c10::ivalue::Tuple::create(std::move(row))));
        }
        return ret;
    } 
    return ret;
}


static void
infer_batch_internal(VecAggState *state, bool ret_float8)
{
    std::string model_path;
    int prcsd_batch_n = state->ins->length;
    
    if(strlen(state->model) == 0){
        ereport(ERROR, (errmsg("model name is empty!")));
    }

    if(!model_manager.GetModelPath(state->model, model_path)){
        model_path.clear();
        ereport(ERROR, (errmsg("model not exist, can't get path!")));
    }

    // 1. 加载模型
    {
        CLOCK_START();

        if(!model_manager.LoadModel(state->model, model_path)){
            model_path.clear();
            ereport(ERROR, (errmsg("load model error")));
        }

        CLOCK_END(load_model);
    }   

    // 2. 设置gpu模式
    if(pg_strcasecmp(state->cuda, "gpu") == 0 && 
       model_manager.SetCuda(model_path)){
        model_path.clear();
        ereport(ERROR, (errmsg("failed to set gpu mode!")));
    }

    std::vector<std::thread> pool;
    std::vector<int> res(prcsd_batch_n, 0);
    std::vector<std::vector<torch::jit::IValue>> input_tensors(prcsd_batch_n, std::vector<torch::jit::IValue>());
    std::vector<std::vector<at::Tensor>> batch_inputs_tmp;
    std::vector<torch::jit::IValue> input_batch_tensor; // batch of tensors
    torch::jit::IValue output;
    std::vector<torch::jit::IValue> outputs; // batch of tuples<tensor|list|tuple> or tensors

    // 3. 输入预处理
    {
        CLOCK_START();

        for (int i = 0; i < prcsd_batch_n; i++) {
            //pool.emplace_back([&, i](){
                Args* in = (Args*)list_nth(state->ins, i);
                res[i] = model_manager.PreProcess(model_path, input_tensors[i], in);
            //});
        }
        //WAIT_AND_CHECK_ERROR("preprocess");

        CLOCK_END(pre);
    }

    // 4. 预测
    {
        CLOCK_START();

        // concat each inputs
        batch_inputs_tmp.resize(input_tensors[0].size());
        for (auto& vecs: input_tensors) {
            for (int i = 0; i < vecs.size(); i++) {
                batch_inputs_tmp[i].emplace_back(vecs[i].toTensor());
            }
        }
        for (auto& one_dim_vecs: batch_inputs_tmp) {
            input_batch_tensor.emplace_back(torch::concat(one_dim_vecs, 0));
        }
            
        // infer model
        if(!model_manager.Predict(model_path, input_batch_tensor, output)) {
            CLEAN_UP_CPP_OBJS();
            ereport(ERROR, (errmsg("%s:predict error!", model_path)));
        }
        CLOCK_END(infer);
    }
    

    // 5. 结果处理
    {
        CLOCK_START();
        try{
            outputs = split_results(output);
            if (outputs.empty()) {
                CLEAN_UP_CPP_OBJS();
                ereport(ERROR, (errmsg("cannot handle the result type from model!")));
            }

            for (int i = 0; i < prcsd_batch_n; i++)
                state->outs = lappend(state->outs, palloc0(sizeof(Args)));

            for (int i = 0; i < prcsd_batch_n; i++) {
                //pool.emplace_back([&, i](){
                    Args* in = (Args*)list_nth(state->ins, i);
                    torch::jit::IValue wrapped_out(outputs[i]);
                    if (ret_float8) {
                        float8& out = ((Args*)list_nth(state->outs, i))->floating;
                        res[i] = model_manager.OutputProcessFloat(model_path, wrapped_out, in, out);
                    } else {
                        std::string result_str;
                        res[i] = model_manager.OutputProcessText(model_path, wrapped_out, in, result_str);   
                        ((Args*)list_nth(state->outs, i))->ptr = pstrdup(result_str.c_str());
                    }
                //});
            }
            //WAIT_AND_CHECK_ERROR("postprocess");
        }catch (const std::exception& e) {
            elog(INFO, "error message:%s", e.what());
        }
        CLOCK_END(post);
    }

    /* update infered batch size */
    state->prcsd_batch_n = prcsd_batch_n;
    state->batch_i++;
}


Datum
predict_batch_accum(PG_FUNCTION_ARGS)
{
    VecAggState*    state;
    MemoryContext old_context;
    Args*           vec;

    state = PG_ARGISNULL(0) ? NULL : (VecAggState *) PG_GETARG_POINTER(0);
    
    /* Create the state data on the first call */
    if (state == NULL)
        state = makeVecAggState(fcinfo);

    old_context = MemoryContextSwitchTo(state->ctx);

    vec = makeVecFromArgs(fcinfo, VECTOR_START_ARG_INDEX, PG_NARGS() - VECTOR_START_ARG_INDEX);
    state->ins = lappend(state->ins, vec);

    MemoryContextSwitchTo(old_context);

    PG_RETURN_POINTER(state);
}

Datum
predict_batch_accum_inv(PG_FUNCTION_ARGS) 
{
    VecAggState*    state;
    
    state = (VecAggState *) PG_GETARG_POINTER(0);

    /* finilize has been called */
    Assert((state != NULL) && (state->prcsd_batch_n != 0));

    if (state->nxt_csr == state->prcsd_batch_n) 
    {
        for (int i = 0; i < state->prcsd_batch_n; i++) 
        {
            state->ins = list_delete_first(state->ins);
            state->outs = list_delete_first(state->outs);
        }
        state->nxt_csr -= state->prcsd_batch_n;
        state->prcsd_batch_n -= state->prcsd_batch_n;
    }

    PG_RETURN_POINTER(state);
}

static Args*
fetch_next_from_predicted_batch(PG_FUNCTION_ARGS, bool ret_float8) 
{
    VecAggState*    state = NULL;
    Args*           ret = NULL;
    MemoryContext   old_context;

    state = PG_ARGISNULL(0) ? NULL : (VecAggState *) PG_GETARG_POINTER(0);

    /* If there were no non-null inputs, return NULL */
    if (state == NULL || list_length(state->ins) == 0)
        return NULL;

    /* do batch infer once if need */
    if (state->nxt_csr >= state->prcsd_batch_n)
    {
        old_context = MemoryContextSwitchTo(state->ctx);
        infer_batch_internal(state, ret_float8);
        MemoryContextSwitchTo(old_context);

        if (debug_print_batch_time)
        {
            int64_t total = state->pre_time + state->infer_time + state->post_time;
            ereport(NOTICE, 
                (errmsg("\nbatch %d:\n"
                        " pre process: %ld ms(%.2f%%)\n"
                        " infer: %ld ms(%.2f%%)\n"
                        " post process: %ld ms(%.2f%%)",
                        state->batch_i, 
                        state->pre_time, (state->pre_time / (float)total) * 100, 
                        state->infer_time, (state->infer_time / (float)total)  * 100, 
                        state->post_time, (state->post_time / (float)total  * 100))));
        }
    }

    /* consume one result */ 
    state = (VecAggState *) PG_GETARG_POINTER(0);
    ret = (Args*)list_nth(state->outs, state->nxt_csr);
    state->nxt_csr++;

    return ret;
}

Datum
predict_batch_final_float(PG_FUNCTION_ARGS)
{
    Args* final_ret = fetch_next_from_predicted_batch(fcinfo, true);

    if (final_ret == NULL)
        PG_RETURN_NULL();

    PG_RETURN_FLOAT8(final_ret->floating);
}

Datum
predict_batch_final_text(PG_FUNCTION_ARGS)
{
    Args* final_ret = fetch_next_from_predicted_batch(fcinfo, false);

    if (final_ret == NULL)
        PG_RETURN_NULL();

    PG_RETURN_TEXT_P(cstring_to_text(static_cast<const char*>(final_ret->ptr)));
}

Datum
enable_print_batch_time(PG_FUNCTION_ARGS)
{   
    debug_print_batch_time = PG_ARGISNULL(0) ? true : PG_GETARG_BOOL(0);
    return (Datum)0;
}

Datum
predict_batch_dummy(PG_FUNCTION_ARGS)
{
    return aggregate_dummy(fcinfo);
}

}