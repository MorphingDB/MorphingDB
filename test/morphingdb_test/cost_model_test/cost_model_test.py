import torch
import json
from torchvision.models import (
    resnet18,
    alexnet,
    googlenet
)
from transformers import (
    AlbertModel,
    RobertaModel
)

import torch.nn as nn
import torch.nn.functional as F
import subprocess
import re
import time
import numpy as np
from typing import Dict, Union
from thop import profile
# from mmengine.analysis import get_model_complexity_info
# from mmengine.analysis.print_helper import _format_size
from ptflops import get_model_complexity_info

class HardwareMonitor:
    """获取硬件带宽信息的工具类"""
    
    @staticmethod
    def get_mem_bandwidth() -> float:
        """获取系统内存带宽(GB/s)"""
        try:
            # 使用dmidecode获取内存信息
            cmd = "sudo dmidecode -t memory | grep Speed"
            output = subprocess.check_output(cmd, shell=True).decode()
            speeds = [int(line.split()[1]) for line in output.splitlines() if "MHz" in line]
            avg_speed_mhz = np.mean(speeds) if speeds else 3200  # 默认DDR4-3200
            
            # 假设双通道64位总线
            bandwidth_GBs = (avg_speed_mhz * 2 * 64 / 8) / 1000
            return bandwidth_GBs * 1e9  # 转换为bytes/sec
            
        except Exception as e:
            print(f"Warning: Could not get memory bandwidth, using default 25.6 GB/s. Error: {str(e)}")
            return 25.6 * 1e9  # 默认DDR4-3200的理论带宽

    @staticmethod
    def get_gpu_bandwidth(device_id=0) -> float:
        """获取GPU显存带宽(GB/s)"""
        try:
            # 使用nvidia-smi获取带宽信息
            cmd = f"nvidia-smi --id={device_id} --query-gpu=memory_info.width --format=csv,noheader,nounits"
            bus_width = int(subprocess.check_output(cmd, shell=True).decode().strip())
            
            cmd = f"nvidia-smi --id={device_id} --query-gpu=clocks.mem --format=csv,noheader,nounits"
            mem_clock = int(subprocess.check_output(cmd, shell=True).decode().strip())
            
            # 计算有效带宽 (GDDR5/GDDR6的倍增系数不同)
            bandwidth = (bus_width * mem_clock * 2) / 8  # 转换为GB/s
            return bandwidth * 1e9  # 转换为bytes/sec
            
        except Exception as e:
            print(f"Warning: Could not get GPU bandwidth, using default 900 GB/s. Error: {str(e)}")
            return 900 * 1e9  # 默认A100的带宽
        


class ModelAnalyzer:
    """PyTorch模型分析工具类"""
    
    @staticmethod
    def get_model_size(model: nn.Module) -> int:
        """计算模型参数量(字节)"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """计算可训练参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CostModel:
    """增强版成本模型"""
    
    def __init__(self, device='cuda:0'):
        self.device = device

        self.hw_monitor = HardwareMonitor()
        self.cpu_flops = self._estimate_cpu_flops()
        print(f"cpu_flops:{self.cpu_flops}")
        self.gpu_flops = self._get_gpu_flops()
        print(f"gpu_flops:{self.gpu_flops}")
        
    def _estimate_cpu_flops(self) -> float:
        """估算CPU单核FLOPs"""
        try:
            # 通过/proc/cpuinfo获取CPU信息
            with open('/proc/cpuinfo') as f:
                info = f.read()
            
            # 提取CPU型号和频率
            model_match = re.search(r"model name\s*:\s*(.+)", info)
            freq_match = re.search(r"cpu MHz\s*:\s*(\d+)", info)
            
            if model_match and freq_match:
                model = model_match.group(1).lower()
                freq_ghz = float(freq_match.group(1)) / 1000
                
                # 常见CPU架构的每周期FLOPs
                if 'xeon' in model:
                    flops_per_cycle = 32  # AVX-512
                elif 'core' in model or 'i3' in model or 'i5' in model or 'i7' in model or 'i9' in model:
                    flops_per_cycle = 16  # AVX2
                else:
                    flops_per_cycle = 8  # 保守估计
                
                return flops_per_cycle * freq_ghz * 1e9
                
        except Exception as e:
            print(f"Warning: Could not estimate CPU FLOPs, using default 50 GFLOPS. Error: {str(e)}")
            return 50e9  # 默认值
        
        return 50e9  # 默认值

    def _get_gpu_flops(self) -> float:
        """获取GPU的理论FLOPs"""
        try:
            if not torch.cuda.is_available():
                return 0
                
            # 获取GPU计算能力
            capability = torch.cuda.get_device_capability(self.device)
            print(f"capability:{capability}")
            cuda_cores = {
                (8,0): 6912,  # A100
                (7,5): 4608,   # T4
                (7,0): 5120,    # V100
                (8,6): 10496    # 3090
            }.get(tuple(capability), 2048)  # 默认值
            
            # 获取GPU频率
            cmd = f"nvidia-smi --id={self.device.split(':')[-1]} --query-gpu=clocks.max.sm --format=csv,noheader,nounits"
            sm_clock = int(subprocess.check_output(cmd, shell=True).decode().strip())
            
            # 计算理论FP32 FLOPs
            return cuda_cores * sm_clock * 2 * 1e6  # 2 FLOPS per core per cycle
            
        except Exception as e:
            print(f"Warning: Could not get GPU FLOPs, using default 10 TFLOPS. Error: {str(e)}")
            return 10e12  # 默认值

    def calculate_cost(self, model, batch_size: int, ops_per_param: float = 2) -> Dict[str, Union[float, str]]:
        """完整成本计算流程"""
        input = torch.randn(4, 3, 10, 10) 
        input_shape = (3, 256, 256)
        # opnames = model._c._get_operator_export_type_name_list()
        # model_flops = len(opnames)
        # params = sum([p.numel() for p in model.parameters()])
        model_macs, params = profile(model[0], inputs= (model[1],))
        model_flops = model_macs * 2
        #model_flops, params = get_model_complexity_info(model, input_shape)
        print(f"model_flops, params:{model_flops},{params}")
        # 获取模型参数
        model_params = ModelAnalyzer.count_parameters(model[0])
        model_size = ModelAnalyzer.get_model_size(model[0])
        
        # 获取硬件参数
        cpu_bandwidth = self.hw_monitor.get_mem_bandwidth()
        gpu_bandwidth = self.hw_monitor.get_gpu_bandwidth()
        print(f"cpu_bandwidth, gpu_bandwidth:{cpu_bandwidth},{gpu_bandwidth}")
        # 计算延迟 (经验值)
        latency = 9  # 9s基础延迟
        
        # 计算成本
        gpu_cost = self._calculate_gpu_cost(
            model_flops=model_flops,
            flops=self.gpu_flops,
            num_rows=batch_size,
            model_size_bytes=model_size,
            cpu_bandwidth=cpu_bandwidth,
            gpu_bandwidth=gpu_bandwidth,
            latency=latency
        )
        
        cpu_cost = self._calculate_cpu_cost(
            model_flops=model_flops,
            flops=self.cpu_flops,
            num_rows=batch_size,
            model_size_bytes=model_size,
            cpu_bandwidth=cpu_bandwidth
        )
        
        # 决定平台
        platform = 'GPU' if gpu_cost < cpu_cost else 'CPU'
        
        return {
            'platform': platform,
            'gpu_cost': gpu_cost,
            'cpu_cost': cpu_cost,
            'model_params': model_params,
            'model_size_MB': model_size / (1024 * 1024),
            'gpu_flops_TFLOPS': self.gpu_flops / 1e12,
            'cpu_flops_GFLOPS': self.cpu_flops / 1e9,
            'gpu_bandwidth_GBs': gpu_bandwidth / 1e9,
            'cpu_bandwidth_GBs': cpu_bandwidth / 1e9
        }

    def _calculate_gpu_cost(self, model_flops: float, flops: float, 
                          num_rows: int, model_size_bytes: int, gpu_bandwidth: float, cpu_bandwidth: float, 
                          latency: float) -> float:
        """GPU成本计算"""
        compute_time = model_flops / flops * num_rows
        transfer_cost = (model_size_bytes / cpu_bandwidth) + (model_size_bytes / gpu_bandwidth) + latency
        return compute_time + transfer_cost

    def _calculate_cpu_cost(self, model_flops: float, flops: float, 
                           num_rows: int, model_size_bytes: int, cpu_bandwidth: float) -> float:
        """CPU成本计算"""
        compute_time = model_flops / flops * num_rows
        transfer_cost = (model_size_bytes / cpu_bandwidth)
        return compute_time + transfer_cost


class LogisticRegressionModel_swarm(torch.nn.Module):
    def __init__(self,input_size, output_size):
        super(LogisticRegressionModel_swarm, self).__init__()
        self.lin1 = nn.Linear(input_size,200)
        self.lin2 = nn.Linear(200,output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x

class LogisticRegressionModel_year(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel_year, self).__init__()
        self.linear = torch.nn.Sequential(
                      torch.nn.Linear(input_dim,output_dim))
                    #   torch.nn.PReLU(),
                      
                    #   torch.nn.Linear(128,64),
                    #   torch.nn.PReLU(),
                      
                    #   torch.nn.Linear(64,32),
                    #   torch.nn.PReLU(),
                     
                    #   torch.nn.Linear(32,output_dim))
    def forward(self, x):
        y_pred = self.linear(x)  #self.linear是callable的，是可调用的对象
        return y_pred

class CTslicesModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        ''' 
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.PReLU(),
            nn.Linear(1000, 500),
            nn.PReLU(),
            nn.Linear(500, 250),
            nn.PReLU(),
            nn.Linear(250, 100),
            nn.PReLU(),
            nn.Linear(100, output_size)   
         )
         '''
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, xb):
        out = self.linear(xb)            
        return out

def cost_model_test():
    model_map = {
        "slice": [CTslicesModel(384, 1), torch.randn(1,384)],
        "swarm": [LogisticRegressionModel_swarm(2400, 1), torch.randn(1, 2400)],
        "year_predict": [LogisticRegressionModel_year(90, 1), torch.randn(1,90)],
        "cifar": [googlenet(), torch.randn(1, 3, 224, 224)],
        "imagenet": [resnet18(), torch.randn(1, 3, 224, 224)],
        "stanford_dogs": [alexnet(), torch.randn(1, 3, 224, 224)],
        "imdb": [AlbertModel.from_pretrained("albert-base-v2"), torch.randint(0, 1000, (1, 128))],
        "sst2": [AlbertModel.from_pretrained("albert-base-v2"), torch.randint(0, 1000, (1, 128))],
        "financial_phrase": [RobertaModel.from_pretrained("roberta-base"), torch.randint(0, 1000, (1, 128))]
    }
    
    # 2. 初始化成本模型
    cost_model = CostModel(device='cuda:0')
    
    # 3. 计算不同batch size的成本
    batch_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    COST_MODEL_TEST_FILE = "result/cost_model_test.json"
    try:
        with open(COST_MODEL_TEST_FILE, 'r') as f_vector:
            # 尝试加载现有数据
            timing_data_vector = json.load(f_vector)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
        timing_data_vector = []
    
    for key, model in model_map.items():
        for bs in batch_sizes:
            start = time.time()
            result = cost_model.calculate_cost(model, bs, ops_per_param=4)
            end = time.time()
            timing_data_vector.append({ "model": key,
                                "time": end - start,
                                "Num rows": bs,
                                "Model Params": result['model_params'], 
                                "Model Size": result['model_size_MB'], 
                                "GPU Cost": result['gpu_cost'],
                                "CPU Cost": result['cpu_cost'], 
                                "Recommended Platform": result['platform']})
        # 写回文件
        with open(COST_MODEL_TEST_FILE, 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)

# 使用示例
if __name__ == "__main__":    
    model_map = {
        "slice": [CTslicesModel(384, 1), torch.randn(1,384)],
        "swarm": [LogisticRegressionModel_swarm(2400, 1), torch.randn(1, 2400)],
        "year_predict": [LogisticRegressionModel_year(90, 1), torch.randn(1,90)],
        "cifar": [googlenet(), torch.randn(1, 3, 224, 224)],
        "imagenet": [resnet18(), torch.randn(1, 3, 224, 224)],
        "stanford_dogs": [alexnet(), torch.randn(1, 3, 224, 224)],
        "imdb": [AlbertModel.from_pretrained("albert-base-v2"), torch.randint(0, 1000, (1, 128))],
        "sst2": [AlbertModel.from_pretrained("albert-base-v2"), torch.randint(0, 1000, (1, 128))],
        "financial_phrase": [RobertaModel.from_pretrained("roberta-base"), torch.randint(0, 1000, (1, 128))]
    }
    
    # 2. 初始化成本模型
    cost_model = CostModel(device='cuda:0')
    
    # 3. 计算不同batch size的成本
    batch_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    COST_MODEL_TEST_FILE = "result/cost_model_test.json"
    try:
        with open(COST_MODEL_TEST_FILE, 'r') as f_vector:
            # 尝试加载现有数据
            timing_data_vector = json.load(f_vector)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或内容不是有效的JSON，初始化为一个空列表
        timing_data_vector = []
    
    for key, model in model_map.items():
        for bs in batch_sizes:
            start = time.time()
            result = cost_model.calculate_cost(model, bs, ops_per_param=4)
            end = time.time()
            timing_data_vector.append({ "model": key,
                                "time": end - start,
                                "Num rows": bs,
                                "Model Params": result['model_params'], 
                                "Model Size": result['model_size_MB'], 
                                "GPU Cost": result['gpu_cost'],
                                "CPU Cost": result['cpu_cost'], 
                                "Recommended Platform": result['platform']})
        # 写回文件
        with open(COST_MODEL_TEST_FILE, 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)