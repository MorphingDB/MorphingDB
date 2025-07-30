import psycopg2
import os
import evadb
from morphingdb_test.config import db_config, imagenet_dataset_path

IMAGE_COUNT_LIST = [10000]

IMAGE_TABLE = 'imagenet_image_table'
IMAGE_VECTOR_TABLE = 'imagenet_image_vector_table'


image_name = os.listdir(imagenet_dataset_path)
image_name = image_name[:10000]


def import_imagenet_mvec_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    # 创建表
    cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE + " (id int, image_vector mvec);")
    conn.commit()

    # 删除表中的行
    cur.execute("delete from " + IMAGE_VECTOR_TABLE + ";")
    conn.commit()

    for index in range(len(image_name)):
        sql = f"INSERT INTO " + IMAGE_VECTOR_TABLE + " (id, image_vector) VALUES ({},{})".format(index+1, "image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '{}')".format(imagenet_dataset_path+image_name[index]))
        #print(sql)
        cur.execute(sql)
    conn.commit()

    conn.close()


def import_imagenet_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    # 创建表
    cur.execute("create table if not exists " + IMAGE_TABLE + " (id int, image_path text);")
    conn.commit()
    
    cur.execute("delete from " + IMAGE_TABLE + ";")
    conn.commit()

    # 插入表
    for index in range(len(image_name)):
        sql = f"INSERT INTO " + IMAGE_TABLE  + " (id, image_path) VALUES ({},'{}')".format(index+1, imagenet_dataset_path+image_name[index])
        #print(sql)
        cur.execute(sql)
    conn.commit()

    conn.close()

def import_evadb_imagenet_dataset():
    # 连接evadb
    cursor = evadb.connect().cursor()
    cursor.query("DROP TABLE IF EXISTS IMAGENET").df()
    for index in range(len(image_name)):
        cursor.query("LOAD IMAGE '{}' INTO IMAGENET".format(imagenet_dataset_path+image_name[index])).df()
    




if __name__ == "__main__":
    import_morphingdb_imagenet_dataset()
    #import_evadb_imagenet_dataset()

