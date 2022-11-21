import torch.optim as optim
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt

epochs = 100000
lr = 1e-4


class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

        # self.conv2d = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        # self.act = nn.ReLU(True)
    def forward(self, x):
        
        a = self.conv2d(x)
        
#         x = self.act(x)
        return x + a
    
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data,0)
    elif classname.find('Linear') != -1:
        nn.init.constant_(model.bias.data,1)

model = simpleNet()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()
n_g = 20

weights1010 = torch.load('F:\KAWM\experiments\ARCNN_0404\models/200000_G.pth')
weights4010 = torch.load('F:\KAWM\experiments\ARCNN_1030\models/latest_G.pth')

# print(weights1010.keys())
# print(weights1010['base.0.weight'])
# print(weights1010['base.1.weight'])
# print(weights1010['last.3.weight'])
weights1010 = weights1010['base.2.weight'].cpu()
# print(weights1011.size())

weights4010 = weights4010['base.2.weight'].cpu()

x = weights1010.reshape(-1,7,7)
y = weights4010.reshape(-1,7,7)

x = x.view([-1,1,7,7])
y = y.view([-1,1,7,7])

# x = weights1010.unsqueeze(0)
# y = weights4010.reshape(-1,9,9).unsqueeze(0)

# x = weights1010.reshape(-1,7,7).unsqueeze(0)
# y = weights4010.reshape(-1,7,7).unsqueeze(0)
# for index in range(x.size(0)):
avg_loss_org = 0
avg_loss_m = 0
count = 0
for index in range(2):
    print(count)
    count += 1 
    x_train = x[index].unsqueeze(0)
    y_train = y[index].unsqueeze(0)

    model.apply(initialize_weights);
    model.train()
    for epoch in range(10001):
        pred = model(x_train)
        optimizer.zero_grad()
        loss = criterion(pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch == 10000:
            model.eval()
            with torch.no_grad():
                pred_x = model(x_train)
                val_loss = criterion(pred_x, y_train).item()
                loss2 = criterion(x_train, y_train).item()
                
                avg_loss_m += val_loss
                avg_loss_org += loss2
                
                best_model_weight = model.state_dict()['conv2d.weight']
                # print('index',index,'epoch:',epoch, 'best',val_loss, 'org_loss:', loss2)
                
    image_root_path = "./features"
    org = os.path.join(image_root_path,'modulated_horizontal')
    if not os.path.exists(org):
        os.makedirs(org, exist_ok = True)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    newx = best_model_weight[0][0].cpu().numpy().copy()
    plt.imshow(newx, cmap='gray')
    plt.savefig(os.path.join(org, str(index)+'_g_'+'.jpg'), bbox_inches='tight')

    m_x = pred_x[0][0].cpu().numpy().copy()
    plt.imshow(m_x, cmap='gray')
    plt.savefig(os.path.join(org, str(index)+'_modulated_'+'.jpg'), bbox_inches='tight')

    gt = y_train[0][0].cpu().numpy().copy()
    plt.imshow(gt, cmap='gray')
    plt.savefig(os.path.join(org, str(index)+'_gt_'+'.jpg'), bbox_inches='tight')
    
    inputa = x_train[0][0].cpu().numpy().copy()
    plt.imshow(inputa, cmap='gray')
    plt.savefig(os.path.join(org, str(index)+'_input_'+'.jpg'), bbox_inches='tight')
print('modulated : {:.7f}'.format(avg_loss_m / count), 'org : {:.7f}'.format(avg_loss_org / count))  


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
        