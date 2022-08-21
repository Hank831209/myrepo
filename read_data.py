import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import cv2
import torch.nn.functional as F
# from PIL import Image
# import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # './Data/hymenoptera_data/train', training data的資料夾
        self.label_dir = label_dir  # '0' ---> label(設為資料夾名稱)
        self.path = os.path.join(self.root_dir, self.label_dir)  # '.Data/hymenoptera_data/train/0'
        self.img_path = os.listdir(self.path)  # 所有圖片檔案名稱的list, ex: [00013035.jpg, ...]

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 第 idx 張圖片的檔案名稱, ex: 00013035.jpg
        img_item_path = os.path.join(self.path, img_name)  # ex: '.Data/hymenoptera_data/train/0/00013035.jpg'
        img = cv2.imread(img_item_path)
        img = self.transform(img)  # 圖片前處理(轉tensor等等)
        label = int(self.label_dir)  # class0 ---> ants, ...
        data = (img, label)
        return data

    def __len__(self):
        return len(self.img_path)

    @staticmethod
    def transform(img):
        # img ---> PIL image
        img = cv2.resize(img, (32, 32))
        trans = transforms.ToTensor()
        img = trans(img)
        return img


if __name__ == '__main__':
    # 提取訓練數據集
    train_dir = r'./Data/hymenoptera_data/train'  # 訓練數據集路徑
    train_dataset = 0
    for step, label in enumerate(os.listdir(train_dir)):
        if step == 0:
            train_dataset = MyData(train_dir, label)
        else:
            train_dataset = train_dataset + MyData(train_dir, label)

    # 提取訓驗證據集
    val_dir = r'./Data/hymenoptera_data/val'  # 驗證數據集路徑
    val_dataset = 0
    for step, label in enumerate(os.listdir(val_dir)):
        if step == 0:
            val_dataset = MyData(val_dir, label)
        else:
            val_dataset = val_dataset + MyData(val_dir, label)

    train_data_size = len(train_dataset)  # 資料集的資料總數
    test_data_size = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True, drop_last=False, num_workers=2)

    # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')  # 設置GPU device

    class Classification(nn.Module):
        def __init__(self):
            super(Classification, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.linear1 = nn.Sequential(
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # 幾分類問題
            )

        def forward(self, x):  # x = x.view(x.size(0), -1)
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            x = self.linear1(x)
            return x

    net = Classification()
    net = net.to(device)  # 調用GPU

    # 損失函數
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)  # 調用GPU
    # 優化器
    learning_rate = 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # training and testing
    writer = SummaryWriter('logs')
    total_train_step = 0  # 記錄畫圖用的x軸座標
    EPOCH = 20
    for epoch in range(EPOCH):
        print("-------第 {} 輪訓練開始-------".format(epoch + 1))
        # 訓練
        net.train()
        for (img, label) in train_loader:
            img = img.to(device)  # 調用GPU
            label = label.to(device)  # 調用GPU
            output = net(img)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 畫圖記錄誤差
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("訓練次數：{}, Loss: {}".format(total_train_step, loss.item()))
                # writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 驗證
        net.eval()
        total_correct = 0
        total_test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for (imgs, labels) in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = net(imgs)
                loss = loss_func(outputs, labels)
                total_test_loss = total_test_loss + loss.item()
                predict_correct_num = (F.softmax(outputs, dim=1).argmax(dim=1) == labels).sum().item()
                total_correct = total_correct + predict_correct_num

        accuracy = total_correct / test_data_size
        print("epoch: {}, 驗證集上的總Loss: {}".format(epoch, total_test_loss))
        print("epoch: {}, 驗證集上的正確率: {}".format(epoch, accuracy))
        writer.add_scalar("test_loss", total_test_loss, epoch)
        writer.add_scalar("test_accuracy", accuracy, epoch + 1)

    # torch.save(net, "net.pkl")  # 存整個網路
    torch.save(net.state_dict(), './Data/net_params.pkl')  # 只存網路參數(官方推薦)
    print("模型已保存")
    writer.close()


