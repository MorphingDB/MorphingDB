import psycopg2
import time
import json
#from morphingdb_test.config import db_config

db_config = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5448",
    "user": "lhh"
}


def api_load_model_test():
    # 初始化结果列表
    results = []

    # 定义要执行的 SQL 查询
    queries = [
        ("resnet18", "/home/lhh/morphingdb/MorphingDB_test/models/resnet18_vec.pt"),
        ("resnet50", "/home/lhh/models/resnet50_resnet50_nabird.pt"),
        ("alexnet", "/home/lhh/morphingdb/MorphingDB_test/models/alexnet_stanford_dogs.pt"),
        ("googlenet", "/home/lhh/morphingdb/MorphingDB_test/models/googlenet_cifar10.pt")
    ]

    for model_name, model_path in queries:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        start = time.time()
        sql = f"select api_load_model('{model_name}','{model_path}');"
        cur.execute(sql)
        end = time.time()

        # 计算执行时间
        execution_time = end - start

        # 将结果存储为字典
        result = {
            "sql": sql,
            "execution_time": execution_time
        }

        # 将字典添加到结果列表
        results.append(result)

        # 关闭数据库连接
        cur.close()
        conn.close()

    # 将结果列表写入 JSON 文件
    with open("result/api_load_model_results.json", "w") as result_file:
        json.dump(results, result_file, indent=4)








if __name__ == "__main__":
    api_load_model_test()