'''
Author: laihuihang laihuihang@foxmail.com
Date: 2024-08-15 21:31:09
LastEditors: laihuihang laihuihang@foxmail.com
LastEditTime: 2024-08-26 10:36:58
FilePath: /morphingdb_test/text_test/imdb/evadb_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import evadb
import time
import json
import os
from morphingdb_test.config import evadb_db_config

TEXT_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TEXT_TEST_FILE = 'result/evadb_financial_phrasebank_test.json'

cursor_create = evadb.connect().cursor()

query = cursor_create.query("""
    DROP DATABASE IF EXISTS postgres_data;
""").df()
sql = f"""
    CREATE DATABASE IF NOT EXISTS postgres_data
    WITH ENGINE = 'postgres',
    PARAMETERS = {evadb_db_config}
"""
print(sql)
query = cursor_create.query(f"""
    CREATE DATABASE IF NOT EXISTS postgres_data
    WITH ENGINE = 'postgres',
    PARAMETERS = {evadb_db_config}
""").df()

# query = cursor.query("""
#     DROP FUNCTION FinanceTest;
# """).df()

query = cursor_create.query("""
    CREATE FUNCTION IF NOT EXISTS FinanceTest
    TYPE  HuggingFace
    TASK 'text-classification'
    MODEL 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
""").df()

def gpu_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for count in TEXT_COUNT_LIST:
        cursor = evadb.connect().cursor()
        start_time = time.time()
        sql = "SELECT FinanceTest(comment) FROM postgres_data.financial_phrasebank_test limit {};".format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(TEXT_TEST_FILE, 'r') as f_image:
                # 尝试加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_image = []

        timing_data_image.append({
            "sql": "gpu " + sql,
            "count": count,
            "total_time": end_time - start_time
        })
        # 写回文件
        with open(TEXT_TEST_FILE, 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)


def cpu_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for count in TEXT_COUNT_LIST:
        cursor = evadb.connect().cursor()
        start_time = time.time()
        sql = "SELECT FinanceTest(comment) FROM postgres_data.financial_phrasebank_test limit {};".format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(TEXT_TEST_FILE, 'r') as f_image:
                # 尝试加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_image = []

        timing_data_image.append({
            "sql": "cpu " + sql,
            "count": count,
            "total_time": end_time - start_time
        })
        # 写回文件
        with open(TEXT_TEST_FILE, 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)


def evadb_financial_phrasebank_test():
    #gpu_test()
    cpu_test()

if __name__ == "__main__":
    gpu_test()
    cpu_test()