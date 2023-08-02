#include "model_manager.h"

#include "spi_connection.h"
#include "md5.h"
#include <cstdio>
#include <unistd.h>


extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "miscadmin.h"
}

//内置的模型注册回调函数可以在构造函数中
ModelManager::ModelManager()
{
    
}

ModelManager::~ModelManager()
{
    
}


bool ModelManager::CreateModel(const std::string model_name, 
                               const std::string model_path, 
                               const std::string discription)
{
    std::string md5_value;

    MD5 md5;
    md5_value = md5.ComputeFileMD5(model_path);

    Oid userId = GetUserId(); 
    char *userName = GetUserNameFromId(userId, false); 

    SPIConnector spi_connector;

    std::string prepare_str = "INSERT INTO model_info \
        (model_name, model_path, create_time, update_time, md5, discription, upload_by) \
        VALUES ($1, $2, now(), now(), $3, $4, $5)";
    
    SPISqlWrapper sql(spi_connector, prepare_str, 5);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name.c_str())) &&
       sql.Bind(2, TEXTOID, CStringGetTextDatum(model_path.c_str())) &&
       sql.Bind(3, TEXTOID, CStringGetTextDatum(md5_value.c_str())) && 
       sql.Bind(4, TEXTOID, CStringGetTextDatum(discription.c_str())) &&
       sql.Bind(5, TEXTOID, CStringGetTextDatum(userName)) &&
       sql.Execute()){
        if(SPI_processed != 1){
           return false;
        }
    }else{
        return false;
    }
    
    return true;
}

bool ModelManager::UpdateModel(const std::string model_name, 
                               const std::string model_path)
{
    std::string md5_value;
    MD5 md5;
    md5_value = md5.ComputeFileMD5(model_path);

    SPIConnector spi_connector;

    std::string prepare_str = "UPDATE model_info \
        SET model_path=$2, md5=$3, update_time=now() \
        WHERE model_name=$1";
    
    SPISqlWrapper sql(spi_connector, prepare_str, 3);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name.c_str())) && 
       sql.Bind(2, TEXTOID, CStringGetTextDatum(model_path.c_str())) &&
       sql.Bind(3, TEXTOID, CStringGetTextDatum(md5_value.c_str())) &&
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }else{
        return false;
    }

    return true;
}

bool ModelManager::DropModel(const std::string model_name)
{
    SPIConnector spi_connector;

    std::string prepare_str = "DELETE FROM model_info \
                               WHERE model_name=$1";

    SPISqlWrapper sql(spi_connector, prepare_str, 1);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name.c_str())) && 
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }else{
        return false;
    }

    return true;
}

bool ModelManager::GetModelPath(const std::string model_name, 
                                std::string& model_path)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT model_path FROM model_info \
                               WHERE model_name=$1";

    SPISqlWrapper sql(spi_connector, prepare_str, 1);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name.c_str())) && 
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }
    // get result
    HeapTuple tuple = SPI_tuptable->vals[0];
    model_path = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);
    
    return true;
}

bool ModelManager::GetModelMd5(const std::string model_path, 
                               std::string& md5)
{
    SPIConnector spi_connector;

    std::string prepare_str = "SELECT md5 FROM model_info \
                               WHERE model_path=$1";

    SPISqlWrapper sql(spi_connector, prepare_str, 1);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_path.c_str())) && 
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }
    
    // 获取select 语句结果
    HeapTuple tuple = SPI_tuptable->vals[0];
    md5 = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);
    

    return true;
}

bool ModelManager::GetModelDeviceType(const std::string model_name,
                            torch::DeviceType& device_type)
{
    std::string model_path;
    if(GetModelPath(model_name, model_path)){
        if(module_handle_.find(model_name) != module_handle_.end()){
            device_type = module_handle_[model_name].second;
            return true;
        }
        return false;
    }
    return false;
}

bool ModelManager::GetLastModelVersion(const std::string model_name, 
                             int16& version)
{
    SPIConnector spi_connector;
    std::string version_str;

    std::string prepare_str = "SELECT version FROM model_info \
                               WHERE model_name=$1 ORDER BY version DESC\
                               LIMIT 1";

    SPISqlWrapper sql(spi_connector, prepare_str, 1);

    if(sql.Bind(1, TEXTOID, CStringGetTextDatum(model_name.c_str())) && 
       sql.Execute()){
        if(SPI_processed != 1){
            return false;
        }
    }

    // 获取select 语句结果
    HeapTuple tuple = SPI_tuptable->vals[0];
    version_str = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);

    version = std::atoi(version_str.c_str()) + 1;

    elog(INFO, "version:%d", version);

    return true;
}

bool ModelManager::LoadModel(std::string model_path)
{
    if(module_handle_.find(model_path) != module_handle_.end()){
        return true;
    }
    // Verify the md5 of the model in the file system and the md5 of the database
    std::string table_md5, file_md5;
    MD5 md5;
    file_md5 = md5.ComputeFileMD5(model_path);
    GetModelMd5(model_path, table_md5);
    if(file_md5 != table_md5){
        return false;
    }

    try {
        torch::jit::script::Module cur_module = torch::jit::load(model_path.c_str());
        module_handle_[model_path].first = cur_module;
        // cpu default
        module_handle_[model_path].second = at::kCPU;
        module_handle_[model_path].first.to(at::kCPU);
        cur_module.eval();
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

bool ModelManager::SetCuda(const std::string& model_path)
{
    if(module_handle_.find(model_path) != module_handle_.end()){
        if(module_handle_[model_path].second == at::kCUDA){
            return true;
        } else {
            if(torch::cuda::is_available()){
                module_handle_[model_path].second = at::kCUDA;
                module_handle_[model_path].first.to(at::kCUDA);
                module_handle_[model_path].first.eval();
                ereport(INFO, (errmsg("libtorch use gpu!")));
                return true;
            }
            return false;
        }
    } else {
        return false;
    }
}

bool ModelManager::IsCuda(const std::string& model_path)
{
    if(module_handle_.find(model_path) == module_handle_.end()){
        return false;
    }
    return module_handle_[model_path].second == at::kCUDA;
}

bool ModelManager::PreProcess(const std::string model_path, 
                              std::vector<torch::jit::IValue>& img_tensor, 
                              Args* args)
{
    if(module_preprocess_functions_.find(model_path) != module_preprocess_functions_.end()){
        //ereport(INFO, (errmsg("user preprocess function")));
        return module_preprocess_functions_[model_path](img_tensor, args);
    }
    //img_tensor = torch::ones({1,3,224,224});
    return false;
}

bool ModelManager::OutputProcessText(const std::string model_path, 
                                     torch::jit::IValue& output_tensor,
                                     Args* args, 
                                     std::string& result)
{
    if(module_outputprocess_functions_text_.find(model_path) != module_outputprocess_functions_text_.end()){
        return module_outputprocess_functions_text_[model_path](output_tensor, args, result);
    }
    return false;
}

bool ModelManager::OutputProcessFloat(const std::string model_path, 
                                      torch::jit::IValue& output_tensor, 
                                      Args* args,
                                      float8& result)
{
    if(module_outputprocess_functions_float_.find(model_path) != module_outputprocess_functions_float_.end()){
        return module_outputprocess_functions_float_[model_path](output_tensor, args, result);
    }

    return false;
}


void ModelManager::RegisterPreProcess(const std::string& model_name, 
                                      PreProcessCallback func)
{
    std::string model_path;
    if(GetModelPath(model_name, model_path)){
        module_preprocess_functions_[model_path] = func;
        return;
    }else{
        ereport(ERROR, (errmsg("model:%s not exist!", model_name.c_str())));
    }
}

void ModelManager::RegisterOutoutProcessFloat(const std::string& model_name, 
                                              OutputProcessFloatCallback func)
{
    std::string model_path;
    if(GetModelPath(model_name, model_path)){
        module_outputprocess_functions_float_[model_path] = func;
        return;
    }else{
        ereport(ERROR, (errmsg("model:%s not exist!", model_name.c_str())));
    }
}

void ModelManager::RegisterOutoutProcessText(const std::string& model_name, 
                                             OutputProcessTextCallback func)
{
    std::string model_path;
    if(GetModelPath(model_name, model_path)){
        module_outputprocess_functions_text_[model_path] = func;
        return;
    }else{
        ereport(ERROR, (errmsg("model:%s not exist!", model_name.c_str())));
    }
}

// bool ModelManager::Predict(const std::string& model_path, 
//                            at::Tensor& input, 
//                            at::Tensor& output)
// {
//     if(module_handle_.find(model_path) == module_handle_.end()){
//         return false;
//     }
//     try {
//         output = module_handle_[model_path].first.forward({input}).toTensor();
//         return true;
//     }
//     catch (const std::exception& e) {
//         return false;
//     }
// }

// bool ModelManager::Predict(const std::string& model_path, 
//                            std::vector<at::Tensor>& input, 
//                            at::Tensor& output)
// {
//     if(module_handle_.find(model_path) == module_handle_.end()){
//         return false;
//     }
//     // try {
//         std::vector<torch::jit::IValue> ivals; 
//         ivals.reserve(input.size());

//         for (const auto& tensor : input) 
//         { 
//             ivals.push_back(tensor); 
//         }


//         output = module_handle_[model_path].first.forward(ivals).toTuple()->elements()[0].toTensor();
//         return true;
//     // }
//     // catch (const std::exception& e) {
//     //     return false;
//     // }
// }

bool ModelManager::Predict(const std::string& model_path,
                 std::vector<torch::jit::IValue>& input, 
                 torch::jit::IValue& output)
{
    if(module_handle_.find(model_path) == module_handle_.end()){
        return false;
    }
    try {
        output = module_handle_[model_path].first.forward(input);
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}