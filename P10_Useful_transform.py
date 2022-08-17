import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

image_path = r"./hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
# print(img_tensor.dtype)  # torch.float32

# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
# print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB
img_norm = trans_norm(img_tensor)
# print(img_norm[0][0][0])  # out = (input - mean)/ std

# Resize
# print(img_tensor.shape)
# Resize the input image to the given size.
#     If the image is torch Tensor, it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
trans_resize = transforms.Resize((512, 512))  # 只改長寬, 只改[..., H, W], 所以輸入為(h, w) 2維
img_tensor_resize = trans_resize(img_tensor)
print(img_tensor_resize.shape)
print('Hello')
# img_ndarray_resize = img_tensor_resize.numpy()  # 也可以轉為ndarray

# Compose, Resize ---2
print(img_tensor.shape)
trans_resize2 = transforms.Resize(512)
trans_Compose = transforms.Compose([trans_resize2, trans_toTensor])
img_resize2 = trans_Compose(img)
print(img_resize2.shape)
