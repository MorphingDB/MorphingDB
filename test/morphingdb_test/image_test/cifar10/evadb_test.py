import evadb
import time
import json
import os
from morphingdb_test.image_test.cifar10.import_dataset import cifar10_import_evadb_dataset


IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IMAGE_TEST_FILE = 'result/evadb_cifar10_test.json'

cursor = evadb.connect().cursor()


query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS Resnet18Huggingface
    TYPE  HuggingFace
    TASK 'image-classification'
    MODEL 'microsoft/resnet-18'
""").df()


def gpu_test(symbol:str):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    for count in IMAGE_COUNT_LIST:
        cursor = evadb.connect().cursor()
        start_time = time.time()
        sql = "SELECT Resnet18Huggingface(data) FROM CIFAR10 limit {};".format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(IMAGE_TEST_FILE.format(symbol), 'r') as f_image:
                # 尝试加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_image = []

        timing_data_image.append({
            "count": count,
            "sql":sql,
            "total_time": end_time - start_time
        })
        # 写回文件
        with open(IMAGE_TEST_FILE.format(symbol), 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)

def cpu_test(symbol:str):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    for count in IMAGE_COUNT_LIST:
        cursor = evadb.connect().cursor()
        start_time = time.time()
        sql = "SELECT Resnet18Huggingface(data) FROM CIFAR10 limit {};".format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(IMAGE_TEST_FILE.format(symbol), 'r') as f_image:
                # 尝试加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_image = []

        timing_data_image.append({
            "count": count,
            "sql":sql,
            "total_time": end_time - start_time
        })
        # 写回文件
        with open(IMAGE_TEST_FILE.format(symbol), 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)

def evadb_cifar_test():
    cifar10_import_evadb_dataset()
    #gpu_test('gpu')
    cpu_test('cpu')

if __name__ == "__main__":
    gpu_test('gpu')
    cpu_test('cpu')