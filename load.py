import torch

print('dd')
pt = torch.load('F:/KAWM/pretrained/QHAN/train_model_1290.pth')

print('dd')
print(pt['network'].keys())