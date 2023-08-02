/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-05-04 22:51:49
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2023-06-30 11:44:37
 * @FilePath: /postgres-DB4AI/src/udf/spi_connection.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "env.h"
#include "spi_connection.h"


SPIConnector::SPIConnector() :
    is_connected_(false),
    is_prepared_(false),
    plan_(NULL)
{
    Connect();
}

SPIConnector::~SPIConnector()
{
    DisConnect();
}

bool SPIConnector::Connect()
{
    if(!is_connected_){
        if(SPI_connect() != SPI_OK_CONNECT){
            return false;
        }
        is_connected_ = true;
    }
    return true;
}

void SPIConnector::DisConnect()
{
    if (is_connected_) {
        if (is_prepared_) {
            SPI_freeplan(plan_);
            plan_ = nullptr;
            is_prepared_ = false;
        }
        SPI_finish();
        is_connected_ = false;
    }
}

bool SPIConnector::IsConnected()
{
    return is_connected_;
}

bool SPIConnector::IsPrepared()
{
    return is_prepared_;
}

bool SPIConnector::Prepare(const std::string& query, 
                           std::vector<Oid>& arg_types)
{
    if(!is_connected_){
        return false;
    }
    if(is_prepared_){
        SPI_freeplan(plan_);
        plan_ = nullptr;
        is_prepared_ = false;
    }
    plan_ = SPI_prepare(query.c_str(), arg_types.size(), arg_types.data());
    if(plan_ == nullptr){
        return false;
    }
    is_prepared_ = true;
    return true;
}

bool SPIConnector::PrepareExecute(std::vector<Datum>& values)
{
    if(!is_connected_){
        return false;
    }
    if(!is_prepared_){
        return false;
    }
    SPI_execp(plan_, values.data(), nullptr, 0);
    return true;
}

bool SPIConnector::Execute(const std::string& query)
{
    if(!is_connected_){
        return false;
    }
    if(is_prepared_){
        SPI_freeplan(plan_);
        plan_ = nullptr;
        is_prepared_ = false;
    }
    SPI_execute(query.c_str(), true, 0);
    return true;
}


SPISqlWrapper::SPISqlWrapper(SPIConnector& conn, 
                             const std::string& sql, 
                             const int parameters):
    pg_conn_(conn),
    sql_(sql)
{
    param_types_.resize(parameters, 0);
    param_values_.resize(parameters, 0);
}

SPISqlWrapper::~SPISqlWrapper()
{
    
}

bool SPISqlWrapper::Bind(int index, 
                         Oid type, 
                         Datum value)
{
    if (index < 1 || index > param_types_.size()) {
        return false;
    }
    param_types_[index-1] = type;
    param_values_[index-1] = value;
    return true;
}

bool SPISqlWrapper::Execute()
{
    //Verify that all values in the vector are filled in
    for(int index=0; index<param_types_.size(); index++){
        if(param_types_[index] == 0){
            return false;
        }
    }
    if(!pg_conn_.IsPrepared()){
        if(!Prepare()){
            return false;
        }
    }
    return pg_conn_.PrepareExecute(param_values_);
}

bool SPISqlWrapper::Prepare()
{
    return pg_conn_.Prepare(sql_, param_types_);
}