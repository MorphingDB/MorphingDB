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

extern "C" {
#include "miscadmin.h"
}


#define BEGIN_DO_IN_CONN() \
    PG_TRY();              \
    {

#define END_DO_IN_CONN(is_in_DisConnect) \
    }                                  \
    PG_CATCH();                        \
    {                                  \
        if (!is_in_DisConnect)           \
        {                              \
            DisConnect();              \
            PG_RE_THROW();             \
        }                              \
    }                                  \
    PG_END_TRY();


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
    // support multi thread, skip stack depth check
    restore_stack_base(NULL);

    if(!is_connected_){
        conn_lock_.lock();
        if(SPI_connect() != SPI_OK_CONNECT){
            return false;
        }
        is_connected_ = true;
    }
    return true;
}

void SPIConnector::DisConnect()
{
    if (!is_connected_)
        return;

    BEGIN_DO_IN_CONN();

    if (is_prepared_) {
        SPI_freeplan(plan_);
        plan_ = nullptr;
        is_prepared_ = false;
    }
    SPI_finish();
    is_connected_ = false;

    END_DO_IN_CONN(true);

    conn_lock_.unlock();
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

    BEGIN_DO_IN_CONN();

    plan_ = SPI_prepare(query.c_str(), arg_types.size(), arg_types.data());
    if(plan_ == nullptr){
        return false;
    }
    is_prepared_ = true;

    END_DO_IN_CONN(false);

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

    BEGIN_DO_IN_CONN();

    SPI_execp(plan_, values.data(), nullptr, 0);
    
    END_DO_IN_CONN(false);

    return true;
}

bool SPIConnector::Execute(const std::string& query)
{
    if(!is_connected_){
        return false;
    }

    BEGIN_DO_IN_CONN();

    if(is_prepared_){
        SPI_freeplan(plan_);
        plan_ = nullptr;
        is_prepared_ = false;
    }
    SPI_execute(query.c_str(), true, 0);
    
    END_DO_IN_CONN(false);

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