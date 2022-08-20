import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

'''
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)  # 可使用官方文件給的公式去求出padding
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)  # input數量如果算不準可以透過改forward算shape去算
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


tudui = Tudui()
print(tudui)
# 驗證神經網路是否可正常運行, 創建一套假數據跑看看會不會報錯
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)
'''


class Tudui2(nn.Module):
    def __init__(self):
        super(Tudui2, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 可使用官方文件給的公式去求出padding
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# Sequential搭建的效果同上, 好處在於forward的時候比較簡潔
tudui2 = Tudui2()
print(tudui2)
input = torch.ones((64, 3, 32, 32))
output = tudui2(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(tudui2, input)  # 畫出神經網路的規劃圖, 包括每個位置的輸入大小等等
writer.close()
