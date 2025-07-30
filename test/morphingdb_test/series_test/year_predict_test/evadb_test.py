import evadb
import time
import json
import os
from morphingdb_test.config import evadb_db_config

IRIS_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
YEAR_PREDICT_TEST_FILE = 'result/evadb_year_predict_test.json'


cursor = evadb.connect().cursor()
# cursor.drop_function("YearPredictClassifier")
# cursor.create_function("YearPredictClassifier", "./morphingdb_test/series_test/year_predict_test/evadb_year_predict.py")


# pg
query = cursor.query("""
    DROP DATABASE IF EXISTS postgres_data;
""").df()
print(query)
query = cursor.query(f"""
    CREATE DATABASE IF NOT EXISTS postgres_data
    WITH ENGINE = 'postgres',
    PARAMETERS = {evadb_db_config}
""").df()
print(query)

# query = cursor.query("""
#     SELECT *
#     FROM postgres_data.iris;
# """).df()
# print(query)



# predict
query = cursor.query("""
    DROP FUNCTION IF EXISTS YearPredictClassifier;
""").df()


values_str_list = [f"value{i} NDARRAY FLOAT32" for i in range(90)]
full_str = ", ".join(values_str_list)
query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS YearPredictClassifier
    INPUT  ({})
    OUTPUT (labels NDARRAY FLOAT32(1))
    TYPE  Classification
    IMPL  './morphingdb_test/series_test/year_predict_test/evadb_year_predict.py';
""".format(full_str)).df()

# query = cursor.query("""
#     SHOW FUNCTIONS;
# """)
# print(query.df())


values_str_list = [f"value{i}" for i in range(90)]
full_str = ", ".join(values_str_list)

def gpu_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    for count in IRIS_COUNT_LIST:
        start_time = time.time()
        sql = """SELECT YearPredictClassifier({})
            FROM postgres_data.year_predict_origin_test limit {};
        """.format(full_str, count)
        cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s")

        try:
            with open(YEAR_PREDICT_TEST_FILE, 'r') as f_image:
                # 加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")
            timing_data_image = []  # 初始化一个空列表，因为文件不存在
            
            # print(timing_data_image)
        timing_data_image.append({"sql": "gpu" + sql, 
                                "count": count, 
                                "total_time": end_time - start_time})
        
        # 将更新后的列表写回文件
        with open(YEAR_PREDICT_TEST_FILE, 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)


def cpu_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    for count in IRIS_COUNT_LIST:
        start_time = time.time()
        sql = """SELECT YearPredictClassifier({})
            FROM postgres_data.year_predict_origin_test limit {};
        """.format(full_str, count)
        cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s")

        try:
            with open(YEAR_PREDICT_TEST_FILE, 'r') as f_image:
                # 加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")
            timing_data_image = []  # 初始化一个空列表，因为文件不存在
            
            # print(timing_data_image)
        timing_data_image.append({"sql": "cpu" + sql, 
                                "count": count, 
                                "total_time": end_time - start_time})
        
        # 将更新后的列表写回文件
        with open(YEAR_PREDICT_TEST_FILE, 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)

def evadb_year_predict_test():
    #gpu_test()
    cpu_test()

if __name__ == "__main__":
    gpu_test()
    cpu_test()