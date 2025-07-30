from collections import OrderedDict

import pandas as pd

from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision

import time
import json
import torch
from sentencepiece import SentencePieceProcessor

from morphingdb_test.config import evadb_sst2_model_path, evadb_spiece_model_path

from transformers import BertTokenizer

TEXT_TEST_FILE = 'result/evadb_muti_query_test.json'



class SST2Test(PytorchAbstractClassifierFunction):

    # def __del__(self):
    #     try:
    #         with open(TEXT_TEST_FILE, 'r') as f_image:
    #             # 尝试加载现有数据
    #             timing_data_image = json.load(f_image)
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
    #         timing_data_image = []

    #     self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
    #     # 将新的记录追加到列表中
    #     if(self.count == 0):
    #         return
    #     timing_data_image.append({"count": self.count, 
    #                             "total_time": 0, 
    #                             "scan_time": 0,
    #                             "load_model_time": self.load_model_time, 
    #                             "pre_time": self.pre_time,
    #                             "infer_time": self.infer_time, 
    #                             "post_time": self.post_time})

    #     print({"count": self.count, 
    #                             "total_time": 0, 
    #                             "scan_time": 0,
    #                             "load_model_time": self.load_model_time, 
    #                             "pre_time": self.pre_time,
    #                             "infer_time": self.infer_time, 
    #                             "post_time": self.post_time})
    #     # 写回文件
    #     with open(TEXT_TEST_FILE, 'w') as f_image:
    #         json.dump(timing_data_image, f_image, indent=4)
        
    @property
    def name(self) -> str:
        return "SST2TestMuti"

    def setup(self):
        try_to_import_torch()
        try_to_import_torchvision()
        import torch
        import torch.nn as nn

        model_urls = {
            "sst2":  evadb_sst2_model_path # noqa
        }

        self.sentencepiece_model = SentencePieceProcessor()

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0
        self.count = 0

    
        def sst2():
            model = torch.load(model_urls["sst2"])
            return model

        start = time.time()
        self.model = sst2()
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start
        self.sentencepiece_model.Load(evadb_spiece_model_path)

    @property
    def labels(self, predict):
        # print("self.load_model_time", self.load_model_time)
        # print("self.pre_time", self.pre_time)
        # print("self.infer_time", self.infer_time)
        # print("self.post_time", self.post_time)
        # print("self.total_time", self.total_time)
        if predict == 0:
            return "消极态度"
        else:
            return "积极态度"

    def transform(self, text):
        start = time.time()
        # sp_model_path = "/home/lhh/pgdl_basemodel_new/model/spiece.model"
        # sp_processor = SentencePieceProcessor()
        # sp_processor.Load(sp_model_path)

        sp_processor = self.sentencepiece_model
        # 对文本进行编码
        tis_int_a = sp_processor.Encode(text)

        # 添加特殊标记
        cls_token_id = sp_processor.PieceToId("[CLS]")
        sep_token_id = sp_processor.PieceToId("[SEP]")
        # tis_int_a.insert(0, cls_token_id)
        # tis_int_a.append(sep_token_id)
        pad_token_id = sp_processor.PieceToId("<pad>")

        # 插入特殊标记并截断到最大长度
        max_length = 128
        tis_int_a = [cls_token_id] + tis_int_a + [sep_token_id]

        tis = tis_int_a[:]
        if len(tis) < max_length: 
            tis += [pad_token_id] * (max_length - len(tis))

        # 创建attention_mask和token_type_ids
        am = [1] * len(tis_int_a)
        ttis = [0] * len(tis_int_a)
        am += [0] * (max_length - len(am))
        ttis += [1] * (max_length - len(ttis))

        # 创建position_ids
        position_ids = torch.arange(0, max_length, dtype=torch.long)

        # 创建PyTorch张量
        token_ids = torch.tensor(tis, dtype=torch.long)
        attention_mask = torch.tensor(am, dtype=torch.long)
        token_type_ids = torch.tensor(ttis, dtype=torch.long)

        # 增加批次维度
        token_ids = token_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        token_type_ids = token_type_ids.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)

        # 假设设备类型是CPU或CUDA，根据实际情况设置
        device_type = torch.device("cpu")
        token_ids = token_ids.to(device_type)
        attention_mask = attention_mask.to(device_type)
        token_type_ids = token_type_ids.to(device_type)
        position_ids = position_ids.to(device_type)
        
        stacked_inputs = torch.stack((token_ids, attention_mask, token_type_ids, position_ids), dim=1)
        end = time.time()
        self.pre_time += (end - start)
        return stacked_inputs

    def forward(self, tensor) -> pd.DataFrame:
        self.count += 1
        start = time.time()
        token_ids, attention_mask, token_type_ids, position_ids = tensor[0]

        token_ids = token_ids.unsqueeze(0).to(torch.device("cpu"))
        attention_mask = attention_mask.unsqueeze(0).to(torch.device("cpu"))
        token_type_ids = token_type_ids.unsqueeze(0).to(torch.device("cpu"))
        position_ids = position_ids.unsqueeze(0).to(torch.device("cpu"))
        end = time.time()
        self.pre_time += (end-start)

        start = time.time()
        self.model = self.model.to(torch.device("cpu"))
        outputs = self.model.forward(token_ids, attention_mask, token_type_ids, position_ids)
        end = time.time()
        self.infer_time += (end-start)

        start = time.time()
        predict = torch.cat(outputs,0).argmax(1).item()
        # 将预测结果转换为DataFrame
        outcome = {"labels": predict}
        end = time.time()
        self.post_time += (end-start)
        labels = pd.DataFrame([outcome], columns=["labels"])

        return labels
