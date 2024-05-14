#include "model_utils.h"
#include "spi_connection.h"

extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "miscadmin.h"
}


bool get_mvec_oid(int32_t& oid)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT oid FROM pg_type WHERE typname='mvec'";
    SPISqlWrapper sql(spi_connector, prepare_str, 0);

    if(sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }

    HeapTuple tuple = SPI_tuptable->vals[0];
    oid = atoi(SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1));
    return true;
}


bool 
get_model_layer_size(const char* model_name, int32_t& layer_size)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT count(*) FROM model_layer_info \
                               WHERE model_name=$1";
    SPISqlWrapper sql(spi_connector, prepare_str, 1);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name)) && 
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }

    HeapTuple tuple = SPI_tuptable->vals[0];
    layer_size = atoi(SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1));

    return true;
}

bool 
get_model_layer_name(const char* model_name, int32_t layer_index, std::string& layer_name)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT layer_name FROM model_layer_info \
                               WHERE model_name=$1 AND layer_index=$2";
    SPISqlWrapper sql(spi_connector, prepare_str, 2);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name)) && 
       sql.Bind(2, INT4OID, Int32GetDatum(layer_index)) &&
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }

    HeapTuple tuple = SPI_tuptable->vals[0];
    layer_name = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);

    return true;
}

bool compare_model_struct(const torch::jit::script::Module& model, const torch::jit::script::Module& base_model)
{
    std::vector<std::pair<std::string, torch::Tensor>> base_model_parameters_vec;
    auto model_modules = model.named_modules();
    auto base_model_modules = base_model.named_modules();

    auto model_parameters = model.named_parameters();
    auto base_model_parameters = base_model.named_parameters();

    if (model_modules.size() != base_model_modules.size() || 
        model_parameters.size() != base_model_parameters.size()) {
        return false;
    }

    for (const auto& pair : base_model_parameters) {
        std::string layer_name = pair.name;
        torch::Tensor parameter = pair.value;
        base_model_parameters_vec.push_back(std::make_pair(layer_name, parameter));
    }

    auto iter = base_model_parameters_vec.begin();
    for (const auto& pair : model_parameters) {
        std::string layer_name = pair.name;
        torch::Tensor model_tensor = pair.value;

        if (layer_name.find("fc") != std::string::npos) {
            continue;
        }
        
        if(iter->first != layer_name || model_tensor.sizes() != iter->second.sizes()){
            return false;
        }
    }
    

    return true;
}

int compare_model_struct(const char* model_path, const char* base_model_path)
{
    torch::jit::script::Module model, base_model;
    try {
        model = torch::jit::load(model_path);
        base_model = torch::jit::load(base_model_path);
    }
    catch (const std::exception& e) {
        return 1;
    }
    std::vector<std::pair<std::string, torch::Tensor>> base_model_parameters_vec;
    std::vector<std::pair<std::string, torch::Tensor>> base_model_buffers_vec;

    auto model_modules = model.named_modules();
    auto base_model_modules = base_model.named_modules();

    auto model_parameters = model.named_parameters();
    auto base_model_parameters = base_model.named_parameters();

    auto model_buffers = model.named_buffers();
    auto base_model_buffers = base_model.named_buffers();

    if (model_modules.size() != base_model_modules.size() || 
        model_parameters.size() != base_model_parameters.size()) {   
        return 2;
    }

    for (const auto& pair : base_model_parameters) {
        std::string layer_name = pair.name;
        torch::Tensor parameter = pair.value;
        base_model_parameters_vec.push_back(std::make_pair(layer_name, parameter));
    }

    for (const auto& pair : base_model_buffers) {
        std::string layer_name = pair.name;
        torch::Tensor buffer = pair.value;
        base_model_buffers_vec.push_back(std::make_pair(layer_name, buffer));
    }

    auto iter = base_model_parameters_vec.begin();
    for (const auto& pair : model_parameters) {
        std::string layer_name = pair.name;
        torch::Tensor model_tensor = pair.value;

        if (layer_name.find("fc") != std::string::npos) {
            continue;
        }
        
        if(iter->first != layer_name || model_tensor.sizes() != iter->second.sizes()){
            return 3;
        }
        iter++;
    }

    iter = base_model_buffers_vec.begin();
    for (const auto& pair : model_buffers) {
        std::string layer_name = pair.name;
        torch::Tensor model_tensor = pair.value;
        
        if(iter->first != layer_name || model_tensor.sizes() != iter->second.sizes()){
            return 3;
        }
        iter++;
    }
    

    return 0;
}

bool 
get_model_layer_parameter(const char* model_name, int32_t layer_index, torch::Tensor& tensor)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT parameter FROM model_layer_info \
                               WHERE model_name=$1 AND layer_index=$2";
    SPISqlWrapper sql(spi_connector, prepare_str, 2);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name)) && 
       sql.Bind(2, INT4OID, Int32GetDatum(layer_index)) &&
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }

    HeapTuple tuple = SPI_tuptable->vals[0];
    bool is_null;
    MVec* vector =  DatumGetMVec(SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &is_null));
    if (is_null) {
        return false;
    }
    tensor = vector_to_tensor(vector);
    return true;
}

bool 
get_model_layer_parameter(const char* model_name, const char* layer_name, torch::Tensor& tensor)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT parameter FROM model_layer_info \
                               WHERE model_name=$1 AND layer_name=$2";
    SPISqlWrapper sql(spi_connector, prepare_str, 2);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name)) && 
       sql.Bind(2, TEXTOID, CStringGetTextDatum(layer_name)) &&
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }

    HeapTuple tuple = SPI_tuptable->vals[0];
    bool is_null;
    MVec* vector =  DatumGetMVec(SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &is_null));
    if (is_null) {
        return false;
    }
    tensor = vector_to_tensor(vector);
    return true;
}

bool 
insert_model_layer_parameter(const char* model_name, const char* layer_name, int32_t layer_index, int32_t oid, MVec* vector)
{
    SPIConnector spi_connector;
    std::string prepare_str = "INSERT INTO model_layer_info \
                               (model_name, layer_name, layer_index, parameter) \
                               VALUES ($1, $2, $3, $4)";

    SPISqlWrapper sql(spi_connector, prepare_str, 4);
    std::string str;
    mvec_to_str(vector, str);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name)) &&
       sql.Bind(2, TEXTOID, CStringGetTextDatum(layer_name)) &&
       sql.Bind(3, INT4OID, Int32GetDatum(layer_index)) && 
       //TODO 这部分因为内核中没有MVec的OID，因此需要先将MVec 转成字符串，之后再转换成MVec进行保存
       sql.Bind(4, oid, (Datum)vector) &&
       sql.Execute()){
        if(SPI_processed != 1){
           return false;
        }
    }else{
        return false;
    }
    
    return true;
}

bool 
delete_model_parameter(const char* model_name)
{
    SPIConnector spi_connector;

    std::string prepare_str = "DELETE FROM model_layer_info \
                               WHERE model_name=$1";

    SPISqlWrapper sql(spi_connector, prepare_str, 1);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name)) && 
       sql.Execute()){
        if(SPI_processed == 0){
            return false;
        }
    }else{
        return false;
    }

    return true;
}

void 
model_parameter_extraction(const char* model_path, 
                                ModelLayer** parameter_list, 
                                int32_t& layer_size)
{
    //uint32 layer_size = 0;
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    }
    catch (const std::exception& e) {
        *parameter_list = NULL;
        //ereport(ERROR, (errmsg("load model failed, error message: %s", e.what())));
    }
    
    auto parms = model.named_parameters();
    auto buffers = model.named_buffers();
    layer_size = parms.size() + buffers.size();
    //ereport(INFO, (errmsg("layer_size:%d", layer_size)));
    *parameter_list = (ModelLayer*)palloc((layer_size) * sizeof(ModelLayer));

    int index=0;
    for(const auto& pair : parms){
        std::string name = pair.name;
        torch::Tensor tensor = pair.value;

        (*parameter_list)[index].layer_name = (char*)palloc((name.size() + 1) * sizeof(char));
        strcpy((*parameter_list)[index].layer_name, name.c_str());

        MVec* vector = tensor_to_vector(tensor);
        (*parameter_list)[index].layer_parameter = vector;

        index++;
    }
    for(const auto& pair : buffers){
        std::string name = pair.name;
        torch::Tensor tensor = pair.value;

        (*parameter_list)[index].layer_name = (char*)palloc((name.size() + 1) * sizeof(char));
        strcpy((*parameter_list)[index].layer_name, name.c_str());

        MVec* vector = tensor_to_vector(tensor);
        (*parameter_list)[index].layer_parameter = vector;

        index++;
    }
}

// void 
// model_parameter_merging(const char* model_name, 
//                         torch::jit::script::Module& model)
// {
//     int32_t layer_size = 0;
//     auto layer_tensor_parm = model.named_parameters();
//     //ereport(INFO, (errmsg("layer_size:%d", layer_tensor_parm.size())));
//     // verify model layer num
//     if(get_model_layer_size(model_name, layer_size)){
//         if(layer_tensor_parm.size() != layer_size){
//             //ereport(ERROR, (errmsg("layer_size:%d, model_layer_size:%d don't match!", layer_tensor_parm.size(), layer_size)));
//             return;
//         }
//     }

//     for(const auto& parm : layer_tensor_parm){
//         torch::Tensor layer_tensor = parm.value.detach();
//         torch::Tensor tensor;
//         if(get_model_layer_parameter(model_name, parm.name.c_str(), tensor)){
//             layer_tensor.copy_(tensor);
//         }
//     }
// }
