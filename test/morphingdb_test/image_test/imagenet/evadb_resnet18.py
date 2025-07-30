# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict

import pandas as pd

from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision

import time
import json
import os
import torch

from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from morphingdb_test.config import evadb_imagenet_model_path

IMAGE_TEST_FILE = 'result/evadb_imagenet_test.json'

class Resnet18Test(PytorchAbstractClassifierFunction):

    def __del__(self):
        try:
            with open(IMAGE_TEST_FILE, 'r') as f_image:
                # 尝试加载现有数据
                timing_data_image = json.load(f_image)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
            timing_data_image = []

        self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
        # 将新的记录追加到列表中
        timing_data_image.append({"count": self.count, 
                                "total_time": 0, 
                                "scan_time": 0,
                                "load_model_time": self.load_model_time, 
                                "pre_time": self.pre_time,
                                "infer_time": self.infer_time, 
                                "post_time": self.post_time})

        print({"count": self.count, 
                                "total_time": 0, 
                                "scan_time": 0,
                                "load_model_time": self.load_model_time, 
                                "pre_time": self.pre_time,
                                "infer_time": self.infer_time, 
                                "post_time": self.post_time})
        # 写回文件
        with open(IMAGE_TEST_FILE, 'w') as f_image:
            json.dump(timing_data_image, f_image, indent=4)
        
    @property
    def name(self) -> str:
        return "Resnet18Test"

    def setup(self):
        try_to_import_torch()
        try_to_import_torchvision()
        import torch
        import torch.nn as nn

        model_urls = {
            "resnet18": evadb_imagenet_model_path  # noqa
        }

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0
        self.count = 0

        def resnet18():
            model = torch.load(model_urls["resnet18"])
            return model

        start = time.time()
        self.model = resnet18()
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

    @property
    def labels(self):
        res = list([str(num+1) for num in range(1000)])
        return res

    def transform(self, images):

        # resnet18 预训练模型使用的规范化参数
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        start = time.time()
        composed = Compose(
            [

                ToTensor(),  # 将PIL图像或NumPy数组转换成FloatTensor
                Normalize(mean=mean, std=std),  # 规范化处理
            ]
        )
        # 确保输入images是RGB三通道
        if images.shape[2] == 3:
            frame = composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)
            end = time.time()
            self.pre_time += (end - start)
            return frame
        else:
            # 如果不是三通道，可能是灰度图，需要转换
            raise ValueError("Input image is not a RGB image")

    def forward(self, frames) -> pd.DataFrame:

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices == '-1':
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        self.count += 1
        outcome = []
        start = time.time()
        frames = frames.to(device)
        self.model = self.model.to(device)
        predictions = self.model(frames)
        end = time.time()
        self.infer_time += (end - start)

        start = time.time()
        for prediction in predictions:
            label = self.as_numpy(prediction.data.argmax())
            outcome.append({"labels": str(label)})

        labels = pd.DataFrame(outcome, columns=["labels"])
        end  = time.time()
        self.post_time += (end - start)

        return labels
