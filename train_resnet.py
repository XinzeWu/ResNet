import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import os
import cv2
import torchvision
import torch
import numpy as np

## 网络构建
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet101()#pretrained=True
    # resnet默认图片输入大小为224*224*3
    def forward(self, x):
        x= self.resnet(x)
        return x
      
net = ResNet().cuda()

##参数设置
class_num = [500,85,37,58,116]
epochs = 10
lr = 0.0001
batch_size = 256
loss_fun = nn.CrossEntropyLoss()
opt_SGD = torch.optim.SGD(net.parameters(), lr=lr)

##数据读取
pic_path = 'E:/university/project/daoshizhi/data2/'
data = []
label = []
for i in range(5):
    pic_name = os.listdir(pic_path + str(i))
    for name in pic_name:
        img = cv2.imread(pic_path + str(i)+ "/" + name)
        data.append(img)
        label.append(i)
        
#数据的保存        
data = np.array(data)
label = np.array(label)
np.save('data',data)
np.save("label",label)

#数据的选取
index = []
total = 0
for i in range(5):
    total += class_num[i]
    for j in range(10):
        index.append(total-j-1)
data = data.reshape((796,3,224,224))
label = label.reshape((796,1))
data1 = data[index]
label1 = label[index]
data = torch.from_numpy(data1)
label = torch.from_numpy(label1)

#网络的训练
for epoch in range(epochs):
        running_loss = 0.0
        opt_SGD.zero_grad()
        pre = net(data.float())
        loss = loss_fun(pre,label.long().squeeze())
        loss.backward()
        opt_SGD.step()
        running_loss += loss.item()
        print("Epoch%03d: Training_loss = %.5f" % (epoch + 1, running_loss))
