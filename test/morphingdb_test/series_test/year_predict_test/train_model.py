import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd


from sklearn.model_selection import train_test_split

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
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







data = pd.read_csv(r"./data/YearPredictionMSD.csv", nrows=50000)


X = data.iloc[:,1:]
Y = data.iloc[:,0]

X = (X - X.mean())/X.std()

X_shuffle = X.sample(frac=1, random_state=0)
Y_shuffle = Y.sample(frac=1, random_state=0)

x_new, x_test, y_new, y_test = train_test_split(X_shuffle, Y_shuffle, test_size=0.2, random_state=0)
dev_per = x_test.shape[0]/x_new.shape[0]
x_train, x_dev, y_train, y_dev = train_test_split(x_new, y_new, test_size=dev_per, random_state=0)

x_train = torch.tensor(x_train.values).float()
y_train = torch.tensor(y_train.values).float()

x_dev = torch.tensor(x_dev.values).float()
y_dev = torch.tensor(y_dev.values).float()

x_test = torch.tensor(x_test.values).float()
y_test = torch.tensor(y_test.values).float()


# # 读取CSV文件
# data = pd.read_csv(r"/data/yearpredict/YearPredictionMSD.csv")

# # 分割数据为训练集和测试集
# train_data = data.iloc[:40000, :]
# test_data = data.iloc[40000:, :]

# # 假设X为特征，Y为目标变量
# X_train = train_data.iloc[:, 1:]
# Y_train = train_data.iloc[:, 0]

# X_test = test_data.iloc[:, 1:]
# Y_test = test_data.iloc[:, 0]

# # 对训练集特征进行标准化处理
# mean = X_train.mean()
# std = X_train.std()

# X_train = (X_train - mean) / std

# # 测试集也使用训练集的均值和标准差进行标准化
# X_test = (X_test - mean) / std

# # 转换数据为PyTorch张量
# x_train = torch.tensor(X_train.values).float()
# y_train = torch.tensor(Y_train.values).float()

# x_test = torch.tensor(X_test.values).float()
# y_test = torch.tensor(Y_test.values).float()

print("x_train.shape[1]", x_train.shape[1])
model = LogisticRegressionModel(x_train.shape[1], 1)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


for i in range(10000):
    y_pred = model(x_train).squeeze()
    loss = loss_function(y_pred, y_train)
    
    optimizer.step()
    optimizer.zero_grad()
    loss.backward()
    
    if i%250 == 0:
        print(i, loss.item())

# 预测测试集
y_pred_test = model(x_test).squeeze()

# 计算预测误差的平方
squared_errors = (y_pred_test - y_test) ** 2

# 计算均方误差（MSE）
mse = squared_errors.mean()

# 计算均方根误差（RMSE）
rmse = mse.sqrt()
print("Test RMSE:", rmse.item())

print("y_pred_test", y_pred_test)
print("y_test", y_test)
#torch.save(model, "year_predict.pth")

example_input = torch.rand(1, x_train.shape[1])
scripted_model = torch.jit.trace(model, example_input)
torch.jit.save(scripted_model, './models/year_predict.pt')