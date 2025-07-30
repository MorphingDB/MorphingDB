import psycopg2
import os
import evadb
from morphingdb_test.config import db_config, cifar10_dataset_path

IMAGE_COUNT_LIST = [1000]

IMAGE_TABLE = 'cifar_image_table'
IMAGE_VECTOR_TABLE = 'cifar_image_vector_table'

# 初始化一个空列表来存储所有文件的路径
image_names = []

# 使用os.walk遍历IMAGE_PRE_PATH及其所有子目录
for root, dirs, files in os.walk(cifar10_dataset_path):
    for file in files:
        # 构建完整的文件路径
        full_path = os.path.join(root, file)
        # 将文件路径添加到列表中
        image_names.append(full_path)



def cifar10_import_mvec_dataset():
    # 连接MorphingDB数据库
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE  + " (id int, image_vector mvec);")
    conn.commit()

    cur.execute("delete from " + IMAGE_VECTOR_TABLE + ";")
    conn.commit()

    for index in range(len(image_names)):
        sql = f"INSERT INTO " + IMAGE_VECTOR_TABLE + " (id, image_vector) VALUES ({},{})".format(index+1, "image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '{}')".format(image_names[index]))
        #print(sql)
        cur.execute(sql)
    conn.commit()
    conn.close()


def cifar10_import_url_dataset():
    # 连接MorphingDB数据库
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("create table if not exists " + IMAGE_TABLE  + " (id int, image_path text);")
    conn.commit()

    cur.execute("delete from " + IMAGE_TABLE + ";")
    conn.commit()

    for index in range(len(image_names)):
        sql = f"INSERT INTO " + IMAGE_TABLE + " (id, image_path) VALUES ({},'{}')".format(index+1, image_names[index])
        #print(sql)
        cur.execute(sql)
    conn.commit()
    conn.close()


def cifar10_import_evadb_dataset():
    # 连接evadb
    cursor = evadb.connect().cursor()
    cursor.query("DROP TABLE IF EXISTS CIFAR10").df()
    for index in range(len(image_names)):
        cursor.query("LOAD IMAGE '{}' INTO CIFAR10".format(image_names[index])).df()