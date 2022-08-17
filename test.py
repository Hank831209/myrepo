# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from PIL import Image
# import os
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid
# import cv2
from read_data import MyData

root_dir = './hymenoptera_data/train'
label_dir_ants = 'ants'
label_dir_bees = 'bees'
ants_dataset = MyData(root_dir, label_dir_ants)
bees_dataset = MyData(root_dir, label_dir_bees)  # img, label
train_dataset = ants_dataset + bees_dataset  # 數據集拼接
img, label = train_dataset[125]
# img.show()
print(len(ants_dataset), len(bees_dataset), len(train_dataset))

