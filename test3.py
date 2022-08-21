import torch
import torch.nn.functional as F

outout = torch.tensor([
    [0.5, 0.7],
    [0.3, 0.4]
])
labels = torch.tensor([1, 1])
# print(outout.argmax(dim=1))  # 橫著找
# print(outout.argmax(dim=0))  # 直著找
# print(torch.max(outout, dim=1))
print(F.softmax(outout, dim=1))
print(F.softmax(outout, dim=1).argmax(dim=1))
print((F.softmax(outout, dim=1).argmax(dim=1) == labels).sum().item())

# prediction = torch.max(F.softmax(out, dim=1), 1)[1]  # 取機率高的那個(類別 ---> index)