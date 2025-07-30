from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os
from typing import Dict
import uvicorn

# 初始化FastAPI应用
app = FastAPI(
    title="PyTorch模型推理服务",
    description="提供PyTorch模型加载和图像推理功能的API服务"
)

# 全局变量存储已加载的模型
loaded_models: Dict[str, torch.nn.Module] = {}

# 定义请求体模型
class LoadModelRequest(BaseModel):
    model_name: str
    model_path: str

class PredictRequest(BaseModel):
    model_name: str
    image_url: str

# 图像预处理转换
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 加载模型接口
@app.post("/load_model/")
async def load_model(request: LoadModelRequest):
    try:
        # 检查模型文件是否存在
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at {request.model_path}")
        
        # 加载模型
        model = torch.load(request.model_path, weights_only=False)
        model = model.to("cuda:0")
        model.eval()  # 设置为评估模式
        
        # 存储到全局字典
        loaded_models[request.model_name] = model
        
        return {
            "status": "success",
            "message": f"Model '{request.model_name}' loaded successfully from {request.model_path}",
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 推理接口
@app.post("/predict/")
async def predict(request: PredictRequest):
    try:
        # 检查模型是否已加载
        if request.model_name not in loaded_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not loaded. Please load it first using /load_model/"
            )
        
        # 获取模型
        model = loaded_models[request.model_name]
        
        # 加载图像并预处理
        # 如果是本地路径，直接打开文件
        if request.image_url.startswith("/"):
            image = Image.open(request.image_url)
        else:
            raise HTTPException(status_code=400, detail="Invalid image URL. Please provide a valid local path.")
        
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0)  # 增加批次维度
        input_tensor = input_tensor.to("cuda:0")  # 将输入张量移动到GPU
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "image_url": request.image_url,
            "predicted_class": predicted_class.item(),
            "confidence": torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取已加载模型列表
@app.get("/loaded_models/")
async def get_loaded_models():
    return {
        "loaded_models": list(loaded_models.keys()),
        "count": len(loaded_models)
    }



@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)