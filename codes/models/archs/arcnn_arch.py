from torch import nn
import imageio
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ARCNN(nn.Module):
    def __init__(self, args):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            # nn.Conv2d(64, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            # nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

    def forward(self, x):
        image_root_path = "./features"
        org = os.path.join(image_root_path,'4010')
        if not os.path.exists(org):
            os.makedirs(org, exist_ok = True)
        x = self.base(x)
        
        # weight = F.conv2d(self.conv1.weight, self.adafm1.transformer.weight, padding=(1, 0), groups=64)
        # weight = nn.Parameter(data=weight, requires_grad=False)
        # self.conv1.weight = weight
            
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        
        # for i in range(self.base[0].weight.size(0)):
        #     a = self.base[0].weight[i]
        #     for j in range(3):
        #         newx = a[j].cpu().numpy().copy()
        #         plt.imshow(newx, cmap='gray')
        #         plt.savefig(os.path.join(org, 'first_'+str(i)+'_'+str(j)+'.jpg'), bbox_inches='tight')
        
        # for i in range(self.base[2].weight.size(0)):
        #     a = self.base[2].weight[i]
        #     for j in range(3):
        #         newx = a[j].cpu().numpy().copy()
        #         plt.imshow(newx, cmap='gray')
        #         plt.savefig(os.path.join(org, 'second_'+str(i)+'_'+str(j)+'.jpg'), bbox_inches='tight')
                
        # for i in range(self.last.weight.size(0)):
        #     a = self.last.weight[i]
        #     for j in range(3):
        #         newx = a[j].cpu().numpy().copy()
        #         plt.imshow(newx, cmap='gray')
        #         plt.savefig(os.path.join(org, 'last_'+str(i)+'_'+str(j)+'.jpg'), bbox_inches='tight')
                
        x = self.last(x)
        return x