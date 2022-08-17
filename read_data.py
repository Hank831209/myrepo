from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
# import numpy as np
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid
# import cv2


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # './hymenoptera_data/train', training data的資料夾
        self.label_dir = label_dir  # 'ants', label為他的資料夾名稱
        self.path = os.path.join(self.root_dir, self.label_dir)  # 圖片的路徑地址
        self.img_path = os.listdir(self.path)  # 所有圖片名稱的list

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # index = idx 的圖片名稱
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


