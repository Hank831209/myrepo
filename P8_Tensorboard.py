from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
# image_path = r"./hymenoptera_data/train/ants_image/6240329_72c01e663e.jpg"
# image_path = r"./hymenoptera_data/train/ants/0013035.jpg"

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()
# img_PIL = Image.open(image_path)
# # img_array = np.array(img_PIL)
# # print(type(img_array))
# # print(img_array.shape)
#
# writer.add_image("train", img_array, 1, dataformats='HWC')
# # y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x", 3*i, i)
#
#
