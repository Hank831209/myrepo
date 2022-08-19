import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2


'''
writer = SummaryWriter('logs')
image_path = r"./hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
print(img_tensor.dtype)  # torch.float32
writer.add_image('ToTensor', img_tensor)
writer.close()



# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
writer = SummaryWriter('logs')
image_path = r"./hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm)
print(img_norm[0][0][0])  # out = (input - mean)/ std
writer.close()


# Resize
writer = SummaryWriter('logs')
image_path = r"./hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)

# print(img_tensor.shape)
# Resize the input image to the given size.
#     If the image is torch Tensor, it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
trans_resize = transforms.Resize((128, 128))  # 只改長寬, 只改[..., H, W], 所以輸入為(h, w) 2維
img_tensor_resize = trans_resize(img)
trans_toTensor = transforms.ToTensor()
img_tensor_resize = trans_toTensor(img_tensor_resize)
writer.add_image('Resize', img_tensor_resize, 2)
print(img_tensor_resize.shape)
# img_ndarray_resize = img_tensor_resize.numpy()  # 也可以轉為ndarray
writer.close()
'''

# Compose, Resize ---2
writer = SummaryWriter('logs')
image_path = r"./hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)

trans_toTensor = transforms.ToTensor()
trans_resize2 = transforms.Resize(512)
# 把所有轉換存在一起的概念, 依序執行
trans_Compose = transforms.Compose([
    trans_resize2,
    trans_toTensor
])
img_resize2 = trans_Compose(img)
writer.add_image('Resize2', img_resize2, 0)
print(img_resize2.shape)
writer.close()
