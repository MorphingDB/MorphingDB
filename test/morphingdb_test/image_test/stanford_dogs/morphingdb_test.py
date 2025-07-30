import psycopg2
import time
import json
import re
import os
from morphingdb_test.config import db_config, stanford_dogs_model_path
from morphingdb_test.image_test.stanford_dogs.import_dataset import (
    import_evadb_stanford_dogs_dataset,
    import_stanford_dogs_dataset,
    import_stanford_dogs_vector_dataset,
    IMAGE_TABLE,
    IMAGE_VECTOR_TABLE
)




IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

IMAGE_VECTOR_TEST_FILE = 'result/stanford_dogs_vector_test.json'
timing_data_image = []
timing_data_vector = []

# parse time
def parse_timing_info(timing_str):
    # 使用正则表达式匹配所需的时间信息
    pattern = r'total: (\d+) ms\n load model: (\d+) ms\((\d+\.\d+)%\)\n pre process: (\d+) ms\((\d+\.\d+)%\)\n infer: (\d+) ms\((\d+\.\d+)%\)\n post process: (\d+) ms\((\d+\.\d+)%\)'
    
    # 使用search方法寻找与正则表达式相匹配的子串
    match = re.search(pattern, timing_str)
    
    if match:
        # 将捕获的组转换为字典
        timing_info = {
            'total_time': int(match.group(1)),
            'load_model_time': int(match.group(2)),
            'load_model_percent': float(match.group(3)),
            'pre_time': int(match.group(4)),
            'pre_percent': float(match.group(5)),
            'infer_time': int(match.group(6)),
            'infer_percent': float(match.group(7)),
            'post_time': int(match.group(8)),
            'post_percent': float(match.group(9)),
        }
        return timing_info
    else:
        # 如果没有找到匹配项，返回None或抛出异常
        return None

def create_model():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("select * from model_info where model_name = 'alexnet_stanford_dogs';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('alexnet_stanford_dogs', '{}', '', '');".format(stanford_dogs_model_path))

    conn.commit()
    conn.close()



def stanford_dogs_test(limit_flag:str, symbol:str = 'cpu'):
    for count in IMAGE_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = "select predict_batch_float8('alexnet_stanford_dogs', '{}', image_vector) over (rows between current row and 31 following) from ".format(symbol) + IMAGE_VECTOR_TABLE + " limit {};".format(count)
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入IMAGE_VECTOR_TEST_FILE
        try:
            with open(IMAGE_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({ "sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})

        # 写回文件
        with open(IMAGE_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)


def stanford_dogs_all_test():
    import_stanford_dogs_vector_dataset()
    print("import stanford_dogs mvec table done")
    import_stanford_dogs_dataset()
    print("import stanford_dogs table done")
    create_model()
    print("create model done")
    stanford_dogs_test('', 'gpu')
    print("stanford_dogs test done")



if __name__ == "__main__":
    create_model()
    stanford_dogs_test('', 'gpu')