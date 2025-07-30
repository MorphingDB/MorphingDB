import psycopg2
import os
from morphingdb_test.config import db_config

IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

pre_path = os.path.dirname(os.path.dirname(__file__))
IMAGE_PRE_PATH = pre_path + '/data/image/image-net/data/'
IMAGE_TABLE = 'image_table_'
IMAGE_VECTOR_TABLE = 'image_vector_table_'


image_name = os.listdir(IMAGE_PRE_PATH)



def import_image_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    # 创建表
    for image_count in IMAGE_COUNT_LIST:
        cur.execute("create table if not exists " + IMAGE_TABLE + str(image_count) + " (id int, image_path text);")
        cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE + str(image_count) + " (id int, image_vector mvec);")
        conn.commit()

    # 删除表中的行
    for image_count in IMAGE_COUNT_LIST:
        cur.execute("delete from " + IMAGE_TABLE + str(image_count) + ";")
        cur.execute("delete from " + IMAGE_VECTOR_TABLE + str(image_count) + ";")
        conn.commit()

    # 插入表
    for image_count in IMAGE_COUNT_LIST:
        for index in range(image_count):
            sql = f"INSERT INTO " + IMAGE_TABLE + str(image_count) + " (id, image_path) VALUES ({},'{}')".format(index+1, IMAGE_PRE_PATH+image_name[index])
            #print(sql)
            cur.execute(sql)
        conn.commit()

    for image_count in IMAGE_COUNT_LIST:
        for index in range(image_count):
            sql = f"INSERT INTO " + IMAGE_VECTOR_TABLE + str(image_count) + " (id, image_vector) VALUES ({},{})".format(index+1, "image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '{}')".format(IMAGE_PRE_PATH+image_name[index]))
            #print(sql)
            cur.execute(sql)
        conn.commit()
    conn.close()


def import_text_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    # 创建表
    for image_count in IMAGE_COUNT_LIST:
        cur.execute("create table if not exists " + IMAGE_TABLE + str(image_count) + " (id int, image_path text);")
        cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE + str(image_count) + " (id int, image_vector mvec);")
        conn.commit()

    # 删除表中的行
    for image_count in IMAGE_COUNT_LIST:
        cur.execute("delete from " + IMAGE_TABLE + str(image_count) + ";")
        cur.execute("delete from " + IMAGE_VECTOR_TABLE + str(image_count) + ";")
        conn.commit()

    # 插入表
    for image_count in IMAGE_COUNT_LIST:
        for index in range(image_count):
            sql = f"INSERT INTO " + IMAGE_TABLE + str(image_count) + " (id, image_path) VALUES ({},'{}')".format(index+1, IMAGE_PRE_PATH+image_name[index])
            #print(sql)
            cur.execute(sql)
        conn.commit()

    for image_count in IMAGE_COUNT_LIST:
        for index in range(image_count):
            sql = f"INSERT INTO " + IMAGE_VECTOR_TABLE + str(image_count) + " (id, image_vector) VALUES ({},{})".format(index+1, "image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '{}')".format(IMAGE_PRE_PATH+image_name[index]))
            #print(sql)
            cur.execute(sql)
        conn.commit()
    conn.close()


if __name__ == "__main__":
    # 连接数据库
    import_image_dataset()
    #import_text_dataset()