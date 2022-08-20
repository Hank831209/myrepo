from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
# import numpy as np
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid
# import cv2


class MyData(Dataset):

    def __init__(self, root_dir, label_dir, transform):
        self.root_dir = root_dir  # './Data/hymenoptera_data/train', training data的資料夾
        self.label_dir = label_dir  # '0' ---> label(設為資料夾名稱)
        # '.Data/hymenoptera_data/train/ants'  ---> 圖片的路徑地址
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 所有圖片檔案名稱的list, ex: [00013035.jpg, ...]
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 第 idx 張圖片的檔案名稱, ex: 00013035.jpg
        # ex: '.Data/hymenoptera_data/train/ants/00013035.jpg'
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)  # PIL image格式讀取該圖片
        img = self.transform(img)  # 轉為Tensor類型
        label = int(self.label_dir)  # class0 ---> ants, ...
        data = (img, label)
        return data

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    # <class 'tuple'>
    # img ---> torch.Size([3, 32, 32]) <class 'torch.Tensor'> torch.float32
    # target ---> 3 <class 'int'>
    root_dir = r'./Data/hymenoptera_data/train'
    label_dir_ants = '0'  # class0 ---> 'ants'
    label_dir_bees = '1'  # class1 ---> 'bees'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    ants_dataset = MyData(root_dir, label_dir_ants, transform)
    bees_dataset = MyData(root_dir, label_dir_bees, transform)  # img, label
    train_dataset = ants_dataset + bees_dataset  # 數據集拼接
    img, label = train_dataset[125]
    # <class 'torch.utils.data.dataset.ConcatDataset'> <class 'tuple'>
    print(type(train_dataset), type(train_dataset[0]))
    print(type(img), img.shape, img.dtype)  # <class 'torch.Tensor'> torch.float32
    print(label, type(label))  # <class 'str'>
    # print(len(ants_dataset), len(bees_dataset), len(train_dataset))


