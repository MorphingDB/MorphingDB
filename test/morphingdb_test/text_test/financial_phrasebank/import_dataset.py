import psycopg2
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from morphingdb_test.config import db_config, financial_phrasebank_dataset_path
import morphingdb_test.text_test.financial_phrasebank.morphingdb as morphingdb


tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

TEXT_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TEXT_TABLE = 'financial_phrasebank_test'
TEXT_VECTOR_TABLE = 'financial_phrasebank_vector_test'

SENTENCE_LIST = []

TXT_LIST = [financial_phrasebank_dataset_path+'/data/Sentences_50Agree.txt', 
            financial_phrasebank_dataset_path+'/data/Sentences_66Agree.txt', 
            financial_phrasebank_dataset_path+'/data/Sentences_75Agree.txt', 
            financial_phrasebank_dataset_path+'/data/Sentences_AllAgree.txt']


def import_financial_phrasebank_mvec_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # create table
    cur.execute("create table if not exists " + TEXT_VECTOR_TABLE + " (comment_vec mvec);")
    conn.commit()

    cur.execute("delete from " + TEXT_VECTOR_TABLE + ";")
    conn.commit()

    # insert vectors
    tensor = tokenizer(SENTENCE_LIST, padding=True, truncation=True, return_tensors="pt")
    input_ids = tensor['input_ids']
    attention_mask = tensor['attention_mask']
    input_ids_list = [input_ids[i].unsqueeze(0) for i in range(input_ids.size(0))]
    attention_mask_list = [attention_mask[i].unsqueeze(0) for i in range(attention_mask.size(0))]

    for i in range(len(input_ids_list)):
        stack_tensor = torch.stack([input_ids_list[i], attention_mask_list[i]], 1)
        mvec_str = morphingdb.tensor_to_mvec(stack_tensor)
        sql = sql = f"INSERT INTO " + TEXT_VECTOR_TABLE + " (comment_vec) VALUES ('{}')".format(mvec_str)
        cur.execute(sql)
        conn.commit()
    conn.close()


def import_financial_phrasebank_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # create table
    cur.execute("create table if not exists " + TEXT_TABLE + " (comment text);")
    conn.commit()

    cur.execute("delete from " + TEXT_TABLE + ";")
    conn.commit()

    # insert sentences
    for txt in TXT_LIST:
        with open(txt, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                part_before_at = line.split('@')[0].strip()  
                SENTENCE_LIST.append(part_before_at)
                sql = f"INSERT INTO " + TEXT_TABLE + " (comment) VALUES ('{}')".format(part_before_at.replace("'","''"))
                cur.execute(sql)
                conn.commit()

    conn.close()
