import torchvision
from torch.utils.data import DataLoader


dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(
    root='dataset',
    train=False,
    transform=dataset_transform,
    download=False
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# 測試數據集中第一張圖片及target
img, target = test_data[0]
print(img.shape)
print(target)

# test_loader尺寸和label
for data in test_loader:
    img, target = data
    print(img.shape)
    print(target)
# torch.Size([4, 3, 32, 32]), [batch_size, C, H, W]
# tensor([9, 5, 9, 3])

