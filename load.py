import torch
from collections import OrderedDict

new_dict = OrderedDict()

load_net_1 = torch.load('F:\KAWM\pretrained\RCAN/RCAN4002_new.pth')
load_net_2 = torch.load('F:\KAWM\pretrained\RCAN/RCAN0240_new.pth')
# load_net_2 = torch.load('./pretrained/HAN/HAN0240.pth')

# print(load_net_1['body.1.body.10.kawm.transformer.weight'])
for k, v in load_net_1.items():
    new_dict[k] = v
for k, v in load_net_2.items():
    new_dict[k] = v

# print(new_dict['body.1.body.10.kawm.transformer.weight'])

torch.save(new_dict, './pretrained\RCAN/merged_RCAN_new.pth')
