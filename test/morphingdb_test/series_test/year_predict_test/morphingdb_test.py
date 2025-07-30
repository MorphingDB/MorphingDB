import psycopg2
import json
import re
import time
import os
from morphingdb_test.config import db_config
from morphingdb_test.series_test.year_predict_test.import_dataset import import_year_predict_mvec_table, import_year_predict_table


# db_config = {
#     "dbname": "postgres",
#     "host": "localhost",
#     "port": "5432",
#     "user": "postgres",
#     "password": "123456"
# }
# db_config = {
#     "dbname": "postgres",
#     "host": "localhost",
#     "port": "5432",
#     "user": "postgres",
#     "password": "123456"
# }

IRIS_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IRIS_TEST_FILE = 'result/year_predict_test_{}.json'
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
    pre_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(pre_path, 'models/year_predict.pt')

    #cur.execute("select drop_model('year_predict');")
    cur.execute("select * from model_info where model_name = 'year_predict';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('year_predict', '{}', '', '');".format(model_path))

    conn.commit()
    conn.close()


def year_predict_test(limit_flag:str, symbol:str = 'cpu'):
    for count in IRIS_COUNT_LIST:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        cur.execute("select register_process();")
        start = time.time()
        sql = "select predict_batch_float8('year_predict', '{}', data) over (rows between current row and 128 following) from year_predict_test limit {};".format(symbol, count)
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

def year_predict_all_test():
    import_year_predict_mvec_table()
    print("import year_predict_mvec_table done")
    import_year_predict_table()
    print("import year_predict_table done")
    create_model()
    print("create model done")
    year_predict_test('', 'cpu')
    print("year_predict_test done")
 
if __name__ == "__main__":
    create_model()
    year_predict_test('mem_2', 'cpu')
