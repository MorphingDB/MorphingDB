from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from evadb.functions.abstract.abstract_function import (
    AbstractFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.decorators.decorators import forward

import time
import json
import os

from morphingdb_test.config import evadb_year_predict_model_path

YEAR_PREDICT_TEST_FILE = 'result/evadb_year_predict_test.json'

class YearPredictClassifier(AbstractFunction):

    @property
    def name(self) -> str:
        return "YearPredictClassifier"

    def setup(self):
        # try_to_import_torch()
        # try_to_import_torchvision()
        import torch
        import torch.nn as nn

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0

        model_urls = {
            "year_predict-classifier": evadb_year_predict_model_path  # noqa
        }

        def year_predict(input_dims=4, n_hiddens=[256, 256], n_class=3, pretrained=None):
            model = torch.load(model_urls["year_predict-classifier"])
            return model
        start = time.time()
        self.model = year_predict()
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[f"value{i}" for i in range(90)],
                column_types = [NdArrayType.FLOAT32 for _ in range(90)],
                column_shapes=[(1,1) for _ in range(90)]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["value0"],
                column_types=[
                    NdArrayType.FLOAT32
                ],
                column_shapes=[(None, None, 1)]
            )
        ],
    )
    def forward(self, input: pd.DataFrame) -> pd.DataFrame:

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices == '-1':
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        
        outcome = []
        input_data = input.values
        print('len(input_data)',len(input_data))
        for row in input_data:
            start = time.time()
            input_tensor = torch.tensor(row, dtype=torch.float32).reshape(1, -1)
            end = time.time()
            self.pre_time += (end - start)

            input_tensor = input_tensor.to(device)
            self.model.to(device)

            start = time.time()
            predictions = self.model(input_tensor)
            end = time.time()
            self.infer_time += (end - start)

            start = time.time()
            label_indices = predictions.argmax(dim=1)
            outcome.append([{"labels": str(label_idx.item())} for label_idx in label_indices])
            end = time.time()
            self.post_time += (end - start)

        self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
        try:
            with open(YEAR_PREDICT_TEST_FILE, 'r') as f_image:
                # 尝试加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_image = []

        # 将新的记录追加到列表中
        # timing_data_image.append({"count": len(input_data), 
        #                         "total_time": 0, 
        #                         "scan_time": 0,
        #                         "load_model_time": self.load_model_time, 
        #                         "pre_time": self.pre_time,
        #                         "infer_time": self.infer_time, 
        #                         "post_time": self.post_time})

        # # 写回文件
        # with open(YEAR_PREDICT_TEST_FILE, 'w') as f_image:
        #     json.dump(timing_data_image, f_image, indent=4)
        return pd.DataFrame(outcome, columns=["labels"])

