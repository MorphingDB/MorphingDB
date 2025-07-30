
import psycopg2
import time
import psutil

from morphingdb_test.config import db_config
from morphingdb_test.image_test.cifar10.import_dataset import IMAGE_TABLE

def auto_ml_test():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]

    # 监控指定 PID 的内存使用情况
    process = psutil.Process(pid)

    # 记录开始时的内存使用情况
    start_mem = process.memory_info().rss / (1024 * 1024)  # 转换为MB

    s_time = time.time()
    cur.execute("select image_classification('{}', 'image_path', 'cifar10', 10);".format(IMAGE_TABLE))
    e_time = time.time()

    # 记录结束时的内存使用情况
    end_mem = process.memory_info().rss / (1024 * 1024)  # 转换为MB

    # 计算内存使用差值
    mem_used = end_mem - start_mem


    print("cost time:{}".format(e_time - s_time))
    print("Memory used: {:.2f} MB".format(mem_used))