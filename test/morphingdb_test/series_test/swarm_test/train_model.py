'''
Author: laihuihang laihuihang@foxmail.com
Date: 2024-08-13 15:30:13
LastEditors: laihuihang laihuihang@foxmail.com
LastEditTime: 2024-08-13 22:12:35
FilePath: /morphingdb_test/swarm_test/train_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self,input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.lin1 = nn.Linear(input_size,200)
        self.lin2 = nn.Linear(200,output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x







dataset = pd.read_csv(r"data/Swarm_Behaviour.csv")
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(columns='Swarm_Behaviour',axis=1).values, 
    dataset['Swarm_Behaviour'].values, 
    test_size=0.2)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


train_y = y_train
test_y = y_test

train_X = torch.tensor(X_train).float()
test_X = torch.tensor(X_test).float()
train_y = torch.tensor(train_y).float()
test_y = torch.tensor(test_y).float()

print('train_X', train_X)
print('train_y', train_y)


model = LogisticRegressionModel(2400,1)

#criterion = torch.nn.BCEWithLogitsLoss()  #损失函数，封装好的逻辑损失函数
#criterion = torch.nn.BCELoss() #二分类
loss_cal = nn.BCELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)   #进行优化梯度下降

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)



# 定义损失函数
epochs = 200
for epoch in range(epochs):

    optimizer.zero_grad() #梯度清零
    y_pred = model(train_X)  #x_data输入数据进入模型中
    loss = loss_cal(y_pred, train_y)
    print(f'number of epoch {epoch}, loss {loss.item()}')
    loss.backward() #反向传播
    optimizer.step()  #优化迭代



# 进行预测
predict_out = model(test_X)
print(predict_out)
predict_y = predict_out.detach().numpy().round()
print(predict_y)

example_input = torch.rand(1, 2400)
scripted_model = torch.jit.trace(model, example_input)
torch.jit.save(scripted_model, 'models/swarm.pt')


print('prediction accuracy', accuracy_score(test_y.cpu().numpy(), predict_y))
print('macro precision', precision_score(test_y.cpu().numpy(), predict_y, average='macro'))
print('micro precision', precision_score(test_y.cpu().numpy(), predict_y, average='micro'))
print('macro recall', recall_score(test_y.cpu().numpy(), predict_y, average='macro'))
print('micro recall', recall_score(test_y.cpu().numpy(), predict_y, average='micro'))