from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import cv2

'''
常見兩種方法
writer.add_scalar()
writer.add_image()
可用下列指令開啟logfile
tensorboard --logdir=logs
tensorboard --logdir=logs --port=6007
'''

'''
# 1. writer.add_scalar()

writer = SummaryWriter("logs")

# 同名稱的圖片例如"y=2x"執行兩次程式的話圖片畫跑掉, 建議刪掉logfile再執行
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
'''

# 1. writer.add_image
writer = SummaryWriter("logs")
# image_path = r"./data/train/ants_image/6240329_72c01e663e.jpg"
image_path = r'./data/train/bees_image/29494643_e3410f0d37.jpg'
# 用opencv直接讀取圖片會是ndarray類型
img_ndarray = cv2.imread(image_path)
print(type(img_ndarray))
print(img_ndarray.shape)  # (369, 500, 3)

# 默認(C, H, W)形式, 也可以改其他形式但要設置dataformats, 可操考相關文檔說明
# writer.add_image("train", img_ndarray, 1, dataformats='HWC')
writer.add_image("train", img_ndarray, 2, dataformats='HWC')
# y = 4x
for i in range(100):
    writer.add_scalar('y=4x', 4 * i, i)
writer.close()

