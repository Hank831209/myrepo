import torchvision
from torch import nn
# train_data = torchvision.datasets.ImageNet("Data/ImageNet", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# print(vgg16_true)

# train_data = torchvision.datasets.CIFAR10('Data/data', train=True, transform=torchvision.transforms.ToTensor(),
#                                           download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print('---'*30)
# print(vgg16_true)
# print('---'*30)
#
print(vgg16_false)
print('---'*30)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


