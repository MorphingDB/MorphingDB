import psycopg2
import time
import json
import re
import os
from morphingdb_test.config import db_config
from morphingdb_test.muti_query.import_dataset import import_muti_query_dataset, import_evadb_muti_query_dataset



TEXT_COUNT_LIST = [10000]

TEXT_VECTOR_TEST_FILE = 'result/muti_query_test.json'
MUTI_MODEL_TEST_FILE = 'result/muti_query_test.json'
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
    pre_path = os.path.dirname(os.path.dirname(__file__))
    print(pre_path)
    text_model_path = os.path.join(pre_path, 'models/traced_albert_vec.pt') 
    image_model_path = os.path.join(pre_path, 'models/resnet18_vec.pt')

    cur.execute("select * from model_info where model_name = 'sst2_vec';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('sst2_vec', '{}', '', '');".format(text_model_path))

    cur.execute("select * from model_info where model_name = 'defect_vec';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('defect_vec', '{}', '', '');".format(image_model_path))
    conn.commit()
    conn.close()

def single_model_test(limit_flag:str):
    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT user_id
            FROM conversation_vector
            WHERE datatime < '2024-06-30'
            AND predict_float('sst2_vec', 'cpu', comment)=0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4) 



def single_model_join_test(limit_flag:str):
    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector cv
                JOIN user_table ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2024-06-30'
                AND predict_float('sst2_vec', 'cpu', cv.comment) = 0
                AND ut.birthday < '2024-06-30';
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)

def single_model_batch_test(limit_flag:str):
    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT predict_batch_float8('sst2_vec','cpu',comment)
            OVER (rows between current row and 15 following)=0
            FROM (
            SELECT * FROM conversation_vector
            WHERE datatime BETWEEN
            '2024-01-01' AND '2024-06-30'
            ) AS subquery;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)  

def muti_model_test(limit_flag:str):
    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        # sql = """
        #     SELECT user_id 
        #             FROM conversation_vector 
        #             WHERE predict_float('sst2_vec', 'cpu', comment)=0
        #             AND predict_float('defect_vec', 'cpu', image)=0;
        # """
        sql = """
            SELECT predict_batch_float8('sst2_vec','cpu',comment)
            OVER (rows between current row and 15 following)=0
            FROM (
                SELECT * FROM conversation_vector
                WHERE predict_float('defect_vec', 'cpu', image)=0
            ) AS subquery;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4) 


def different_data_distribution():
    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector cv
                JOIN user_table ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2025-01-01'
                AND predict_float('sst2_vec', 'gpu', cv.comment) = 0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)

    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector_balance cv
                JOIN user_table_balance ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2025-01-01'
                AND predict_float('sst2_vec', 'gpu', cv.comment) = 0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)


    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector_30_balance cv
                JOIN user_table_30_balance ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2025-01-01'
                AND predict_float('sst2_vec', 'gpu', cv.comment) = 0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)


    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector cv
                JOIN user_table ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2025-01-01'
                AND predict_float('sst2_vec', 'cpu', cv.comment) = 0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)

    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector_balance cv
                JOIN user_table_balance ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2025-01-01'
                AND predict_float('sst2_vec', 'cpu', cv.comment) = 0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)


    for count in TEXT_COUNT_LIST:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # 使用conn执行数据库操作
        cur.execute("select register_process();")
        start = time.time()
        sql = """
            SELECT cv.user_id
                FROM conversation_vector_30_balance cv
                JOIN user_table_30_balance ut ON cv.user_id = ut.user_id
                WHERE cv.datatime < '2025-01-01'
                AND predict_float('sst2_vec', 'cpu', cv.comment) = 0;
        """
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        print(end - start)
        #print(cur.fetchall()[0][0])
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # 读取和追加写入TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'r') as f_vector:
                # 尝试加载现有数据
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_vector = []

        # 将新的记录追加到列表中
        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})
        

        # 写回文件
        with open(TEXT_VECTOR_TEST_FILE.format("different_data_distribution"), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)

def muti_query_all_test():
    import_muti_query_dataset()
    print("import muti query dataset done")
    create_model()
    print("create model done")
    single_model_test('')
    single_model_batch_test('')
    single_model_join_test('')
    muti_model_test('')
    different_data_distribution()

if __name__ == "__main__":
    create_model()
    # single_model_test('mem_2')
    # single_model_batch_test('mem_2')
    # single_model_join_test('mem_2')
    # muti_model_test('mem_2')
    different_data_distribution()