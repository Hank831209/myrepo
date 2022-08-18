import torchvision


# 預設回傳為PIL.Image.Image預設為格式
download = False
trans_set = torchvision.datasets.CIFAR10(
    root='dataset',
    train=True,
    download=download
)
test_set = torchvision.datasets.CIFAR10(
    root='dataset',
    train=False,
    download=download
)
'''
# 可使用Debug mode去看console
print(test_set[0])
print(test_set.classes)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# (<PIL.Image.Image image mode=RGB size=32x32 at 0x23805742190>, 3)
img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show()
'''
# download = True
# # 把數據要transforms一次寫一起, 用list包起來即可依序transforms
# dataset_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])
# trans_set = torchvision.datasets.CIFAR10(
#     root='dataset',
#     train=True,
#     transform=dataset_transform,
#     download=download
# )
# test_set = torchvision.datasets.CIFAR10(
#     root='dataset',
#     train=False,
#     transform=dataset_transform,
#     download=download
# )
# print(test_set[1])  # tensor
