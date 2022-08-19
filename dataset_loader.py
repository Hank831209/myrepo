import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(
    root='dataset',
    train=False,
    transform=dataset_transform,
    download=False
)
# drop_last=True 除不盡就不取啦
test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False,  # 是否要打散重新選
    drop_last=True,  # 除不盡就不取啦
    num_workers=0
)

# 測試數據集中第一張圖片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('logs')
# test_loader尺寸和label
for epoch in range(2):
    step = 0
    for data in test_loader:
        img, target = data
        writer.add_images('Epoch:_{}_False'.format(epoch), img, step)  # dataformats="NCHW"
        step = step + 1
        # print(img.shape)
        # print(target)
        # torch.Size([4, 3, 32, 32]), [batch_size, C, H, W]
        # tensor([9, 5, 9, 3])
writer.close()

