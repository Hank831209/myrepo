import torchvision
from torch.utils.data import DataLoader

download = False
# 把數據要transforms一次寫一起, 用list包起來即可依序transforms
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(
    root='dataset',
    train=False,
    transform=dataset_transform,
    download=download
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# # 測試數據集中第一張圖片及target
# img, target = test_data[0]
# print(img.shape)
# print(target)

#
for data in test_loader:
    img, target = data
    print(img.shape)
    print(target)

