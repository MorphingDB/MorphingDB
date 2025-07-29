CREATE OR REPLACE FUNCTION load_base_model(OUT void)
    AS 'MODULE_PATHNAME', 'load_base_model' LANGUAGE C STRICT;

CREATE TABLE IF NOT EXISTS ai_operator(operator text primary key, discription text);
insert into ai_operator values('Add','两个张量逐元素相加');
insert into ai_operator values('Subtract','两个张量逐元素相减');
insert into ai_operator values('Multiply','逐元素相乘');
insert into ai_operator values('Divide','逐元素相除');
insert into ai_operator values('Pow','逐元素求幂');
insert into ai_operator values('Exp','逐元素计算指数');
insert into ai_operator values('Abs','逐元素计算绝对值');
insert into ai_operator values('Sqrt','逐元素计算平方根');
insert into ai_operator values('ReLU','f(x) = max(0, x)');
insert into ai_operator values('Sigmoid','f(x) = 1 / (1 + exp(-x))');
insert into ai_operator values('Softmax','将输入归一化为概率分布，常用于分类任务的输出层。');
insert into ai_operator values('Softplus','f(x) = log(1 + exp(x))');
insert into ai_operator values('MatMul','矩阵乘法');
insert into ai_operator values('Linear','y = Wx + b');
insert into ai_operator values('Convolution','在输入数据上滑动卷积核，提取特征');
insert into ai_operator values('Max Pooling','取局部区域的最大值');
insert into ai_operator values('Average Pooling','取局部区域的平均值');
insert into ai_operator values('Layer Normalization','对每一层的输入进行归一化');
insert into ai_operator values('MSE','L = (y_pred - y_true)^2');
insert into ai_operator values('Gradient Calculation','计算损失函数对参数的梯度');
insert into ai_operator values('L1 Loss','L = |y_pred - y_true|');
insert into ai_operator values('L2 Loss','L = (y_pred - y_true)^2');

CREATE OR REPLACE FUNCTION api_load_model(  
    model_name cstring,
    model_path cstring
)  
RETURNS boolean
AS 'MODULE_PATHNAME', 'api_load_model'  
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION api_predict(  
    model_name cstring,
    image_url cstring
)  
RETURNS boolean
AS 'MODULE_PATHNAME', 'api_predict'  
LANGUAGE C STRICT;

-- insert into base_model_info values('resnet18', 'd17e88193d63653e785ccd1e8314caee', '/home/lhh/models/resnet18_resnet18_imagenet.pt');
-- insert into base_model_info values('alexnet', '722de753346c94d79e7397a28c6f5674', '/home/lhh/models/alexnet_alexnet_imagenet.pt');
-- insert into base_model_info values('googlenet', '6206b310ee0c21ecdb07516645b5c686', '/home/lhh/models/googlenet_googlenet_imagenet.pt');
-- insert into base_model_info values('resnet50', '2e654b8cc31f235b0bc60bd1852517c4', '/home/lhh/models/resnet50_resnet50_voc2007.pt');