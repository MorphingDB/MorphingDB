/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-04-24 18:15:57
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2023-05-08 09:52:02
 * @FilePath: /postgres-DB4AI/src/udf/connection.cpp
 * @Description: 
 */
#include "env.h"
#include "connection.h"
#include "postgres_ext.h"
#include <cstdio>
#include <cstdlib>



PostgresConnector::PostgresConnector():
    conn_(nullptr)
{
    
}


PostgresConnector::~PostgresConnector()
{
    if(conn_ != nullptr){
        PQfinish(conn_);
        conn_ = nullptr; 
    }
}

bool  PostgresConnector::ConnectToDB(const std::string& connstr)
{
    conn_ = PQconnectdb(connstr.c_str());
    ConnStatusType res = PQstatus(conn_);
    if(res != CONNECTION_OK){
        last_error_.error_code = res;
        last_error_.error_message = PQerrorMessage(conn_);
        return false;
    }
    return true;
}

bool  PostgresConnector::ConnectToDB(const std::string& host, const std::string& user, const std::string& password, 
                                    const std::string& database, const std::string& port)
{
    conn_ = PQsetdbLogin(host.c_str(), port.c_str(), nullptr, nullptr, 
                         database.c_str(), user.c_str(), password.c_str());
    ConnStatusType res = PQstatus(conn_);
    if(res != CONNECTION_OK){
        last_error_.error_code = res;
        last_error_.error_message = PQerrorMessage(conn_);
        return false;
    }
    return true;
}

inline bool PostgresConnector::IsConnected()
{
    if(conn_ == nullptr){
        last_error_.error_code = CONNECTION_BAD;
        last_error_.error_message = "postgres not connected!";
        return false;
    }
    ConnStatusType res = PQstatus(conn_);
    if(res != CONNECTION_OK){
        last_error_.error_code = res;
        last_error_.error_message = PQerrorMessage(conn_);
        return false;
    }
    return true;
}

bool  PostgresConnector::Begin()
{
    if(!IsConnected()){
        return false;
    }
    int ret = true;
    PGresult *res = PQexec(conn_,"BEGIN");
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        ret = false;
    }
    PQclear(res);
    res = nullptr;
    return ret;
}

bool  PostgresConnector::Commit()
{
    if(!IsConnected()){
        return false;
    }
    int ret = true;
    PGresult *res = PQexec(conn_,"COMMIT");
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        ret = false;
    }
    PQclear(res);
    res = nullptr;

    return ret;
}

bool  PostgresConnector::Rollback()
{
    if(!IsConnected()){
        return false;
    }
    int ret = true;
    PGresult *res = PQexec(conn_,"ROLLBACK");
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        ret = false;
    }
    PQclear(res);
    res = nullptr;

    return ret;
}


bool  PostgresConnector::SetCharset(const std::string& charset)
{
    return PQsetClientEncoding(conn_, charset.c_str());
}

void  PostgresConnector::DisConnect()
{
    if(conn_ != nullptr){
        PQfinish(conn_);
        conn_ = nullptr;
    }
}

PGconn* PostgresConnector::GetConnection() const
{
    return conn_;
}

LastError PostgresConnector::GetLastError() const
{
    return last_error_;
}

bool  PostgresConnector::Execute(const std::string& sql, int& tup_num)
{
    bool ret = true;
    PGresult* res = PQexec(conn_, sql.c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK && PQresultStatus(res) != PGRES_TUPLES_OK)
    {
        ret = false;
    }
    tup_num = PQntuples(res);
    PQclear(res);
    res = nullptr;
    return ret;
}

PGresult*  PostgresConnector::Prepare(const std::string& stmt_name, const std::string& sql, 
                  int nparams, const Oid* param_types)
{
    return PQprepare(conn_, stmt_name.c_str(),
                     sql.c_str(), nparams, param_types);
}

bool  PostgresConnector::PrepareExecute(const std::string& stmt_name, int nparams, const char* const* param_values, 
                         const int* param_lengths, const int* param_formats, int result_format, int& tup_num)
{
    bool ret = true;
    PGresult* res = PQexecPrepared(conn_, stmt_name.c_str(), nparams, param_values, 
                                   param_lengths, param_formats, result_format);
    if (PQresultStatus(res) != PGRES_COMMAND_OK && PQresultStatus(res) != PGRES_TUPLES_OK)
    {
        ret = false;
    }
    tup_num = atoi(PQcmdTuples(res));
    PQclear(res);
    res = nullptr;
    return ret;
}


SqlWrapper::SqlWrapper(PostgresConnector& conn):
    pg_conn_(conn)
{

}


SqlWrapper::SqlWrapper(PostgresConnector& conn, const std::string& sql):
    pg_conn_(conn),
    sql_(sql)
{
    int n = 0;
    for (size_t i = 0; i < sql.length(); i++) {
        if (sql[i] == '$') {
            n++;
        }
    }
    param_types_.resize(n, 0);
    for (int i = 0; i < n; i++) {
        param_types_[i] = 0; // 设置为 0，表示未知类型
    }
}

SqlWrapper::~SqlWrapper()
{

}

// prepare bind
bool SqlWrapper::Bind(int index, const char* value, int length, int format)
{
    if (index < 1 || index > param_types_.size()) {
        return false;
    }
    param_values_.push_back(value);
    param_lengths_.push_back(length);
    param_formats_.push_back(format);
    return true;
}

bool  SqlWrapper::Bind(int index, const std::string& value)
{
    return Bind(index, value.c_str(), value.length(), 0);
}

bool  SqlWrapper::Bind(int index, int value)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", value);
    return Bind(index, buf, strlen(buf), 0);
}

bool  SqlWrapper::Bind(int index, double value)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", value);
    return Bind(index, buf, strlen(buf), 0);
}


void  SqlWrapper::Prepare()
{
    int nparams = param_types_.size();
    const Oid* param_types = param_types_.data();
    PGresult* res = pg_conn_.Prepare("", sql_, nparams, param_types);
    //stmt_name_ = PQprepareStmtName(res);
    PQclear(res);
    res = nullptr;
}

bool  SqlWrapper::Execute(int& tup_num)
{
    if(stmt_name_.empty()){
        Prepare();
    }
    return pg_conn_.PrepareExecute(stmt_name_, param_values_.size(), param_values_.data(), 
                                   param_lengths_.data(), param_formats_.data(), 0, tup_num);
}