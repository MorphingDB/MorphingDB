import evadb
import time
import json
import os
from morphingdb_test.image_test.stanford_dogs.import_dataset import import_evadb_stanford_dogs_dataset

IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IMAGE_TEST_FILE = 'result/evadb_stanford_dogs_test.json'

cursor = evadb.connect().cursor()



query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS AlexnetStanford
    INPUT  (data NDARRAY FLOAT32(ANYDIM))
    OUTPUT (labels NDARRAY FLOAT32(10))
    TYPE  Classification
    IMPL  './morphingdb_test/image_test/stanford_dogs/evadb_alexnet_stanford_dog.py';
""").df()


def gpu_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    for count in IMAGE_COUNT_LIST:
        start_time = time.time()
        sql = """
            SELECT AlexnetStanford(data)
            FROM STANFORD limit {};
        """.format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(IMAGE_TEST_FILE, 'r') as f_image:
                # 加载现有数据
                timing_data_image = json.load(f_image)

            # 遍历列表
            for item in timing_data_image:
                # 查找count为100且total_time与scan_time都为零的项
                print(item)
                if item["count"] == count and item["total_time"] == 0 and item["scan_time"] == 0:
                    # 修改total_time和scan_time
                    item["total_time"] = end_time - start_time
                    item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
                    item["sql"] = "gpu" + sql
                    # 由于找到了需要的项，可以结束循环
                    break
            
            print(timing_data_image)
            # 将更新后的列表写回文件
            with open(IMAGE_TEST_FILE, 'w') as f_image:
                json.dump(timing_data_image, f_image, indent=4)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")


def cpu_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    for count in IMAGE_COUNT_LIST:
        start_time = time.time()
        sql = """
            SELECT AlexnetStanford(data)
            FROM STANFORD limit {};
        """.format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(IMAGE_TEST_FILE, 'r') as f_image:
                # 加载现有数据
                timing_data_image = json.load(f_image)

            # 遍历列表
            for item in timing_data_image:
                # 查找count为100且total_time与scan_time都为零的项
                print(item)
                if item["count"] == count and item["total_time"] == 0 and item["scan_time"] == 0:
                    # 修改total_time和scan_time
                    item["total_time"] = end_time - start_time
                    item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
                    # 由于找到了需要的项，可以结束循环
                    item["sql"] = "cpu" + sql
                    break
            
            print(timing_data_image)
            # 将更新后的列表写回文件
            with open(IMAGE_TEST_FILE, 'w') as f_image:
                json.dump(timing_data_image, f_image, indent=4)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")


def evadb_stanford_dogs_test():
    import_evadb_stanford_dogs_dataset()
    gpu_test()
    #cpu_test()

if __name__ == "__main__":
    gpu_test()
    cpu_test()