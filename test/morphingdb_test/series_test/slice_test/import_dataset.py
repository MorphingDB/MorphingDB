'''
Descripttion: 
version: 
Author: LHH
Date: 2024-07-06 21:05:22
LastEditors: LHH
LastEditTime: 2024-07-06 21:33:52
'''
import pandas as pd
import psycopg2
import os
from morphingdb_test.config import db_config, slice_dataset_path



# pre_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# DATA_FILENAME = pre_path + "/data/series/slice/slice_localization_data.csv"
dataframe = pd.read_csv(slice_dataset_path)
value_columns = [col for col in dataframe.columns if col.startswith('value')]


def import_slice_mvec_table():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("drop table if exists slice_test;")
    cur.execute("create table slice_test ( \
                    data mvec, \
                    res float4 \
                );")
    
    formatted_rows = []
    for index, row in dataframe.iterrows():
        # 从当前行中提取value列的值
        values = row[value_columns].tolist()
        # 将值列表转换为字符串形式
        values_str = '[' + ', '.join(str(value) for value in values) + ']'
        # 将结果添加到列表中
        cur.execute("insert into slice_test values('{}', {} );".format(values_str+'{1,384}', row['reference']))
        if(index % 10 == 0):
            conn.commit()
    conn.close()
    


def import_slice_table():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    value_columns.insert(0, 'patientId')
    value_columns.append('reference')
    data_types = ['float4' for _ in value_columns] 

    delete_table_sql = "DROP TABLE IF EXISTS slice_origin_test;"
    cur.execute(delete_table_sql)
    
    create_table_sql = 'CREATE TABLE slice_origin_test (\n  '
    for i, column in enumerate(value_columns):
        create_table_sql += f"{column} {data_types[i]},\n"
    create_table_sql = create_table_sql.rstrip(',\n') + '\n);'
    cur.execute(create_table_sql)
    cur.execute("COPY slice_origin_test FROM '{}' WITH (FORMAT csv, HEADER true, DELIMITER ',');".format(slice_dataset_path))
    conn.commit()

    conn.close()


if __name__ == "__main__":
    import_slice_mvec_table()
    import_slice_table()


