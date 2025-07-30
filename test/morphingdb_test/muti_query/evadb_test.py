import evadb
import time
import json

from morphingdb_test.config import evadb_db_config
from morphingdb_test.muti_query.import_dataset import import_evadb_muti_query_dataset

IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IMAGE_TEST_FILE = 'result/evadb_muti_query_test.json'

cursor = evadb.connect().cursor()

query = cursor.query("""
    DROP DATABASE IF EXISTS postgres_data;
""").df()
print(query)
query = cursor.query(f"""
    CREATE DATABASE IF NOT EXISTS postgres_data
    WITH ENGINE = 'postgres',
    PARAMETERS = {evadb_db_config}
""").df()


query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS Resnet18TestMuti
    INPUT  (data NDARRAY (3, 500, 375))
    OUTPUT (labels NDARRAY FLOAT32(3))
    TYPE  Classification
    IMPL  './morphingdb_test/muti_query/evadb_resnet18.py';
""").df()


query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS SST2TestMuti
    INPUT  (comment TEXT)
    OUTPUT (labels NDARRAY FLOAT32(3))
    TYPE  Classification
    IMPL  './morphingdb_test/muti_query/evadb_sst2.py';
""").df()


# for count in IMAGE_COUNT_LIST:
#     start_time = time.time()
#     res = cursor.query("""
#         SELECT user_id
#         FROM postgres_data.conversation_url
#         WHERE  Resnet18TestMuti(image)=0
#         AND SST2TestMuti(comment)=0;
#     """.format(count)).df()
#     print(res)
#     end_time = time.time()
#     print("cost time:", end_time - start_time, "s") 

#     try:
#         with open(IMAGE_TEST_FILE, 'r') as f_image:
#             # 加载现有数据
#             timing_data_image = json.load(f_image)

#         # 遍历列表
#         for item in timing_data_image:
#             # 查找count为100且total_time与scan_time都为零的项
#             print("11111", item["count"], item["total_time"], item["scan_time"])
#             if item["count"] == count and item["total_time"] == 0 and item["scan_time"] == 0:
#                 print("22222", item["count"], item["total_time"], item["scan_time"])
#                 # 修改total_time和scan_time
#                 item["total_time"] = end_time - start_time
#                 item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
#                 # 由于找到了需要的项，可以结束循环
#                 break
        
#         print("timing_data_image", timing_data_image)
#         # 将更新后的列表写回文件
#         with open(IMAGE_TEST_FILE, 'w') as f_image:
#             json.dump(timing_data_image, f_image, indent=4)

#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         print(f"An error occurred: {e}")

def muti_model_test():
    start_time = time.time()
    sql = """
        SELECT user_id
        FROM conversation_origin
        WHERE SST2TestMuti(comment)=0
        AND Resnet18TestMuti(image)=0;
    """
    res = cursor.query(sql).df()
    print(res)
    end_time = time.time()
    print("cost time:", end_time - start_time, "s") 

    try:
        with open(IMAGE_TEST_FILE, 'r') as f_image:
            # 加载现有数据
            timing_data_image = json.load(f_image)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        timing_data_image = []
    
    timing_data_image.append(
        {
            "sql": sql,
            "cost time": end_time - start_time
        }
    )
    # 将更新后的列表写回文件
    with open(IMAGE_TEST_FILE, 'w') as f_image:
        json.dump(timing_data_image, f_image, indent=4)


def single_model_test():
    start_time = time.time()
    sql = """
        SELECT *
        FROM conversation_origin
        WHERE datetime < '2024-06-30'
        AND SST2TestMuti(comment)=0;
    """
    res = cursor.query(sql).df()
    print(res)
    end_time = time.time()
    print("cost time:", end_time - start_time, "s") 

    try:
        with open(IMAGE_TEST_FILE, 'r') as f_image:
            # 加载现有数据
            timing_data_image = json.load(f_image)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        timing_data_image = []
    
    timing_data_image.append(
        {
            "sql": sql,
            "cost time": end_time - start_time
        }
    )
    # 将更新后的列表写回文件
    with open(IMAGE_TEST_FILE, 'w') as f_image:
        json.dump(timing_data_image, f_image, indent=4)

def evadb_muti_query_test():
    import_evadb_muti_query_dataset()
    muti_model_test()
    single_model_test()

if __name__ == "__main__":
    single_model_test()
    muti_model_test()