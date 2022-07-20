
"""# 导入python包"""

import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

def per_data(train_data):
    train_data = list(train_data)
    train = []
    for i in range(len(train_data)):
        train_data1 = train_data[i]
        train1 = []
        for c in train_data1:
            x = np.array(c)
            x1 = np.real(x)  # 实数
            x2 = np.imag(x)  # 虚数
            train1.append(x1)
            train1.append(x2)
        train.append(np.array(train1))
    train = np.array(train)
    return train

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
eng.cd('./data')
train = eng.data(10000)
train_data = train[0]
train_label1 = train[1]

test = eng.data(1000)
test_data = test[0]
test_label1 =test[1]



train_number = len(train_data)
test_number = len(test_data)

train = per_data(train_data)
train_label = per_data(train_label1)

test = per_data(test_data)
test_label = per_data(test_label1)


train = torch.tensor(train).float()
train_label = torch.tensor(train_label).float()

test = torch.tensor(test).float()
test_label = torch.tensor(test_label).float()

train_ds = TensorDataset(train, train_label)
val_ds = TensorDataset(test, test_label)

bs = 100

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('use:',device)

"""# V1 构建模型 普通模型"""


class Aotuencoder(nn.Module):
    def __init__(self, ):
        super(Aotuencoder, self).__init__()
        self.encoder = nn.Sequential(
          nn.Linear(184, 150),
          nn.ReLU(),
          nn.Linear(150, 120),
          nn.ReLU(),
          nn.Linear(120, 100),
          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
          nn.Linear(100, 120),
          nn.ReLU(),
          nn.Linear(120, 150),
          nn.ReLU(),
          nn.Linear(150, 184)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


aotuencoder =Aotuencoder()
aotuencoder = aotuencoder.to(device)


"""## 分配损失函数和优化器"""

optimzer = torch.optim.Adamax(aotuencoder.parameters(), lr=0.01)
loss_func = nn.MSELoss().to(device)

"""## 定义训练循环"""

# model, opt = get_model()
epochs = 100
best_loss = 100

train_angle_err = []
test_angle_err = []

for epoch in range(epochs):
    running_loss = 0.0
    # for xb, yb in train_dl:
    for i, data in enumerate(train_dl):
        xb = data[0]
        xb = xb.cuda()
        label = data[1]
        label = label.cuda()
        # print(xb.shape,yb.shape)
        pred = aotuencoder(xb)
        
        # print(pred.shape)
        loss = loss_func(pred, label)
        # print(loss,pred[:5],yb[:5])
        loss.backward()
        optimzer.step()
        optimzer.zero_grad()

    # 查看loss情况，每50个batch查看一次
        running_loss += loss.item()
    print(f'[{epoch + 1}] train_loss: {running_loss / train_number:.5f}')
    train_angle_err.append(running_loss / train_number)

        # break
    aotuencoder.eval()
    with torch.no_grad():
        test_running_loss = 0.0
        for i, data1 in enumerate(val_dl):
            xb1 = data1[0]
            xb1 = xb1.to(device)
            label1 = data1[1]
            label1 = label1.cuda()
            # print(xb.shape,yb.shape)
            pred = aotuencoder(xb1)
            # print(pred.shape)
            loss1 = loss_func(pred, labelc1)
            test_running_loss+= loss1.item()
        val_loss =test_running_loss/test_number
        print(f'[{epoch + 1}] test_loss: {test_running_loss / test_number :.5f}')
        test_angle_err.append(test_running_loss / test_number)

    if val_loss<best_loss:
      best_loss = val_loss
      PATH ='./model/best_model.pth'
      # aotuencoder.state_dict()
      torch.save(aotuencoder.state_dict(), PATH)
      print('save best model!')

plt.plot(train_angle_err, label='train_loss')
plt.plot(test_angle_err, label='validate_loss')
plt.title('model train loss and validate loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('out.jpg', dpi=200)
plt.show()
plt.close()
print('done')
