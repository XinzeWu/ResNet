# ResNet
车辆焊接问题

文章链接：https://blog.csdn.net/qq_44647796/article/details/109339214
# 前言

最近要判断一个项目的可行性，项目是一个7分类问题，数据的标签分布是[500,85,37,58,116,8,19]

数据是极度不均衡的，数据量也不够大，好在导师和师兄都比较好，所以我先探索下这个项目可行性

相关代码我已经上传到了GitHub上：
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 20行代码

```python
import torch.nn as nn
import torchvision
import torch
net = torchvision.models.resnet101()
epochs = 1000
lr = 0.001
loss_fun = nn.CrossEntropyLoss()
opt_SGD = torch.optim.SGD(net.parameters(), lr=lr)
data = data.cuda()
label = label.cuda()
for epoch in range(epochs):
        running_loss = 0.0
        opt_SGD.zero_grad()
        pre = net(data.float())
        loss = loss_fun(pre,label.long().squeeze())
        loss.backward()
        opt_SGD.step()
        running_loss += loss.item()
        print("Epoch%03d: Training_loss = %.5f" % (epoch + 1, running_loss))
```
一行不多、
一行不少、

那么接下来来解释下：
# 一、ResNet是什么？

我直接放一个大神的链接吧

[传送门——ResNet详解](https://blog.csdn.net/u013181595/article/details/80990930)



# 二、训练步骤
## 1.引入库
我是采用pytorch的框架，ResNet采用的是101层，最好是采用.cuda()进行加速

<font color=#999AAA >代码如下（示例）：

```python
import torch.nn as nn
import torchvision
import torch
# resnet默认图片输入大小为224*224*3
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=False)  
        #pretrained=True代表使用预训练的模型
    def forward(self, x):
        x= self.resnet(x)
        return x
```

其实duck不必这么麻烦，如下代码也可以

```python
import torch.nn as nn
import torchvision
import torch
resnet = torchvision.models.resnet101()

```

为什么要搞一个class呢？

因为ResNet的默认输入是3\*224\*224的，所以前面可能需要一个卷积操作把数据变换下

**如果你不关心数据的处理，直接想看网络的训练，请跳过第二步**


## 2.处理数据

<font color=#999AAA >代码如下（示例）：


把我的图片转成灰度图，并且统一大小，方便后续的输入
```python
pic_path = your_path
save_path = your_path
for i in range(7):
    pic_name = os.listdir(pic_path + str(i))
    count = 0
    for j in pic_name:
        img = cv2.imread(pic_path + str(i) + "/" + j ,cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path+str(i)+"/" + "{}.png".format(count), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count=count + 1
```
随后，我直接储存成npy格式

```python
import os
import cv2
pic_path = your_path
data = []
label = []
for i in range(5):
    pic_name = os.listdir(pic_path + str(i))
    for name in pic_name:
        img = cv2.imread(pic_path + str(i)+ "/" + name)
        data.append(img)
        label.append(i)
data = np.array(data)
label = np.array(label)
np.save('data',data)
np.save("label",label)
```
## 3.写一个网络训练函数

```python
data = np.load("data.npy")
label = np.load("label.npy")
data = data.reshape((796,3,224,224))
label = label.reshape((796,1))
data = torch.from_numpy(data) #转成tensor才能输入网络训练
label = torch.from_numpy(label)
```
训练网络，最开始的时候不要搞什么策略，上来就要让他过拟合，这样的话来判断一个网络到底能不能用

如果你直接搭建一个很深层的网络都不能学到很好的效果，那么可以考虑：
- 放弃
- 换一个项目
- 数据增强
- 网络更复杂、层数更深

讲真的，为啥把放弃放在第一呢？后面说
如果你的电脑足够强，那么可以把大量的数据直接丢进去，如果不够强，建议使用  [华为的云平台](https://www.huaweicloud.com/)

训练初期：
```python
epochs = 1000
lr = 0.001
loss_fun = nn.CrossEntropyLoss()
opt_SGD = torch.optim.SGD(net.parameters(), lr=lr)
data = data.cuda()
label = label.cuda()
for epoch in range(epochs):
        running_loss = 0.0
        opt_SGD.zero_grad()
        pre = net(data.float())
        loss = loss_fun(pre,label.long().squeeze())
        loss.backward()
        opt_SGD.step()
        running_loss += loss.item()
        print("Epoch%03d: Training_loss = %.5f" % (epoch + 1, running_loss))
```
运行截图

![初期截图](https://img-blog.csdnimg.cn/20201028203624813.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0NjQ3Nzk2,size_16,color_FFFFFF,t_70#pic_center)
等到损失下降到1左右，那么

```python
for epoch in range(epochs):
    if epoch > 10 and flag == 1:
        flag = 0
        opt_SGD = torch.optim.SGD(net.parameters(), lr=lr*0.001)
    running_loss = 0.0
    opt_SGD.zero_grad()
    pre = net(data.float())
    loss = loss_fun(pre,label.long().squeeze())
    loss.backward()
    opt_SGD.step()
    running_loss += loss.item()
    print("Epoch%03d: Training_loss = %.5f" % (epoch + 1, running_loss))
```
我加入了一个学习率的降低
这个学习率可以稍稍展开说下：
**刚开始的时候：lr = 0.001**
**loss=1.多的时候，可以调节为：lr = lr * 0.1**
**等到看损失有几乎不变了，可以再乘以0.1**
**等到乘以的非常小了，loss还是很大，怎么办？**
**注意：敲黑板！！！**
**可以调高学习率，冲一冲，试一试，再调低**
因为网络此时可能训练到了一个局部最优的过程，所以调大学习率可以冲出最优，有人发明了设置了余弦衰减
不过我喜欢手调

最后运行截图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201028204433545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0NjQ3Nzk2,size_16,color_FFFFFF,t_70#pic_center)

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 总结
- 我在华为云平台上跑了一个下午，loss函数从7最后调节到1.05
- 我根据自己的经验判断，这个项目有点难度
- 数据是神经网络的食物，目前的食物有点吃不饱，食不饱、力不足
- 在5分类的情况下（我舍弃了最少的两类数据），使用101层的深度残差网络，仍然无法收敛
- 那么可以做的：
	- 数据增强，采用旋转、调节透明度、平移等方式 
	- 扩大数据集，使得数据分布更加均匀

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

