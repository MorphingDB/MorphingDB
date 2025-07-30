import psycopg2
import json
import re
import time
import os
from morphingdb_test.config import db_config, slice_model_path
from morphingdb_test.series_test.slice_test.import_dataset import import_slice_mvec_table, import_slice_table

COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IRIS_TEST_FILE = 'result/slice_test_{}.json'
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
 

def create_model(model_path: str):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    #pre_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    #model_path = os.path.join(pre_path, 'slice.pt')

    cur.execute("select * from model_info where model_name = 'slice';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('slice', '{}', '', '');".format(model_path))

    conn.commit()
    conn.close()


def slice_test(limit_flag:str, symbol:str = 'cpu'):
    for count in COUNT_LIST:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        cur.execute("select register_process();")
        start = time.time()
        sql = "select predict_batch_float8('slice', '{}', data) over (rows between current row and 31 following) from slice_test limit {};".format(symbol, count)
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        try:
            with open(IRIS_TEST_FILE.format(limit_flag), 'r') as f_vector:

                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):

            timing_data_vector = []


        timing_data_vector.append({"sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})


        with open(IRIS_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)


def slice_all_test():
    import_slice_mvec_table()
    print("import slice mvec table done")
    import_slice_table()
    print("import slice table done")
    create_model(slice_model_path)
    print("create slice model done")
    slice_test('', 'cpu')
    print("slice test done")


if __name__ == "__main__":
    create_model()
    slice_test('mem_2', 'cpu')
    #slice_test('mem_2', 'gpu')
