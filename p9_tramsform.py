from torchvision import transforms  # 按住ctrl點它可以看原始碼
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2


img_path = r'./hymenoptera_data/train/ants/0013035.jpg'
# img = Image.open(img_path)
img = cv2.imread(img_path)

# writer = SummaryWriter('logs')
tensor_trans = transforms.ToTensor()  # 創建一個ToTensor這個class的物件, 可讀PIL Image or np.array
tensor_img = tensor_trans(img)  # 調用ToTensor這個class的 __call__方法, 轉為tensor
# writer.add_image('Tensor_img', tensor_img)
# writer.close()



