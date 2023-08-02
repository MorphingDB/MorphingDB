/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-05-04 18:20:29
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2023-06-30 11:41:56
 * @FilePath: /postgres-DB4AI/src/udf/spi_connection.h
 * @Description: 
 */
#pragma once
#ifndef _SPI_CONNECTION_H_
#define _SPI_CONNECTION_H_

#include <iostream>
#include <string>
#include <vector>


extern "C" {
#include "executor/spi.h"
}

//extern "C" {
//}



// pg内部SPI连接类
class SPIConnector {
public:
    SPIConnector();
    ~SPIConnector();

    /**
     * @description: 创建SPI 连接
     * @event: 
     * @return {*}
     */    
    bool Connect();

    /**
     * @description: 关闭SPI 连接
     * @event: 
     * @return {*}
     */    
    void DisConnect();

    /**
     * @description: 返回连接状态
     * @event: 
     * @return {*}
     */    
    bool IsConnected();

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool IsPrepared();

    /**
     * @description: prepare语句准备
     * @event: 
     * @param {string&} query
     * @return {*}
     */    
    bool Prepare(const std::string& query, 
                 std::vector<Oid>& arg_types);


    bool PrepareExecute(std::vector<Datum>& values);
    bool Execute(const std::string& query);

private:
    bool              is_connected_; // 连接标识
    bool              is_prepared_;  // prepare标识
    SPIPlanPtr        plan_;         
};


class SPISqlWrapper{
public:
    SPISqlWrapper(SPIConnector& conn, 
                  const std::string& sql, 
                  const int parameters);

    ~SPISqlWrapper();

    /**
     * @description: 
     * @event: 
     * @param {int} index
     * @param {Oid} type
     * @param {Datum} value
     * @return {*}
     */    
    bool Bind(int index, 
              Oid type, 
              Datum value);
    
    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool Execute();

private:
    bool Prepare();

private:
    SPIConnector&             pg_conn_;
    std::string               sql_;             //sql语句
    std::vector<Oid>          param_types_;     
    std::vector<Datum>        param_values_;
};


#endif