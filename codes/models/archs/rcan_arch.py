import torch
from torch import nn
from collections import OrderedDict
import numpy as np

from models.archs import common
from utils import util

# Channel Attention (CA) Layer
class CALayer(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    def forensic(self, x):

        inner_forensic_data = {}

        y = self.avg_pool(x)
        inner_forensic_data['inner_vector'] = self.conv_du[1](self.conv_du[0](y)).cpu().data.numpy().squeeze()
        y = self.conv_du(y)

        inner_forensic_data['mask_multiplier'] = y.cpu().data.numpy().squeeze()

        return x * y, inner_forensic_data


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        self.kawm = KAWM(n_feat)
        
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = x[0]
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            if name == '0' and type(midlayer).__name__ == 'Conv2d':
                res = self.kawm(res, x[1])
        #res = self.body(x).mul(self.res_scale)
        res += x[0]
        return res, x[1]

    def forensic(self, x):
        res = x
        conv_data = []
        for module in self.body:
            if isinstance(module, CALayer):
                res, forensic_data = module.forensic(res)
            else:
                res = module.forward(res)
                if isinstance(module, nn.Conv2d):
                    conv_data.append(module.weight.detach().cpu().numpy().flatten())

        forensic_data['conv_flat'] = np.hstack(np.array(conv_data))
        forensic_data['pre-residual'] = res
        forensic_data['pre-residual-flat'] = res.cpu().numpy().flatten()
        res += x
        forensic_data['post-residual'] = res
        forensic_data['post-residual-flat'] = res.cpu().numpy().flatten()
        return res, forensic_data


# Residual Group (RG)
class ResidualGroup(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = x[0]
        for name, midlayer in self.body._modules.items():
            if type(midlayer).__name__ == 'RCAB':
                res, _ = midlayer((res, x[1]))
            else:
                res = midlayer(res)
        # res = self.final_body(res)
        
        res += x[0]
        return res, x[1]

    def forensic(self, x):
        res = x
        forensic_data = []
        for module in self.body:
            if isinstance(module, RCAB):
                res, RCAB_data = module.forensic(res)
                forensic_data.append(RCAB_data)
            else:
                res = module.forward(res)
        res += x
        return res, forensic_data


global layer 
layer = 0
class KAWM(nn.Module):
      
    def __init__(self, in_channel, kernel_size=3):

        super(KAWM, self).__init__()
        
        self.transformer = nn.Conv2d(in_channel, in_channel, (3, 1), bias=False,
                                     padding=(1, 0), groups=in_channel, padding_mode='replicate')
        self.transformer_v = nn.Conv2d(in_channel, in_channel, (1, 3), bias=False,
                                     padding=(0, 1), groups=in_channel, padding_mode='replicate')
        
        self.kernel_transformer_v = nn.Sequential(
            nn.Conv2d(1, in_channel*2, 2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel*2, in_channel, 2, padding=0, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        
        self.kernel_transformer = nn.Sequential(
            nn.Conv2d(1, in_channel*2, 2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel*2, in_channel, 2, padding=0, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        
        constant_init(self.transformer, val=0)
        constant_init(self.transformer_v, val=0)
        
    def forward(self, x, kernel):
        b, c,_,_ = x.size()
        # dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # x = torch.rot90(x, 1, [2,3])
        
        global layer 
        layer += 1
        
        kernel = kernel.reshape(b, 1, 21, 21)
        # kernel = torch.rot90(kernel, -1, [2,3])
        y = self.kernel_transformer(kernel) 
        moduled_x = self.transformer(x) * y 
        
        util.save_tenosor2_img(y[:,:,0,0].transpose(1,0), str(layer)+'_h')
        
        y = self.kernel_transformer_v(kernel) 
        
        # util.save_tenosor2_img(y[:,:,0,0].transpose(1,0), str(layer)+'_v')
        
        moduled_y = self.transformer_v(x) * y 
        
        
        return  x + moduled_x + moduled_y
        # return  x + moduled_x
        # return x = x + moduled_y 
        # return x 

        # return res + x
import torch.nn.functional as F

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x
   
# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, args, n_resblocks=20, n_resgroups=10, n_feats=64, in_feats=3, out_feats=3, scale=4, reduction=16,
                 res_scale=1.0):
        super(RCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        modules_head = [common.default_conv(in_feats, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(common.default_conv, n_feats, kernel_size, reduction,
                          act=act, res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)]

        modules_body.append(common.default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, out_feats, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, kernel):
        dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # x = rot_img(x, np.pi/2, dtype)
        # x = rot_img(x, -np.pi/2, dtype)
        
        x = self.head(x)
        res = x
        for name, midlayer in self.body._modules.items():
            if type(midlayer).__name__ == 'ResidualGroup':
                res, _ = midlayer((res, kernel))
            else:
                res = midlayer(res)
        res += x
        x = self.tail(res)
        
        # x = rot_img(x, -np.pi/2, dtype)
        
        return x

    def forensic(self, x, *args, **kwargs):
        x = self.head(x)
        data = OrderedDict()
        res = x
        for index, module in enumerate(self.body):
            if isinstance(module, ResidualGroup):
                res, res_forensic_data = module.forensic(res)
                for rcab_index, rcab_forensic_data in enumerate(res_forensic_data):
                    data['R%d.C%d' % (index, rcab_index)] = rcab_forensic_data
            else:
                res = module.forward(res)
        res += x
        x = self.tail(res)
        return x, data

    def reset_parameters(self):
        # TODO: Find out how to do this!
        pass

class SRMD(nn.Module):
    """
    Based on implementation in https://github.com/cszn/KAIR/
    Defaults to noise-free model here.
    """
    def __init__(self, in_nc=18, out_nc=3, nc=128, nb=12, scale=4, act_mode='R', upsample_mode='pixelshuffle', **kwargs):
        """
        # ------------------------------------
        in_nc: channel number of input, default: 3+15
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        upscale: scale factor
        act_mode: batch norm + activation function; 'BR' means BN+ReLU
        upsample_mode: default 'pixelshuffle' = conv + pixelshuffle
        # ------------------------------------
        """
        super(SRMD, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = upsample_block(nc, out_nc, mode=str(scale), bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    #    def forward(self, x, k_pca):
    #        m = k_pca.repeat(1, 1, x.size()[-2], x.size()[-1])
    #        x = torch.cat((x, m), 1)
    #        x = self.body(x)

    def forward(self, x):

        x = self.model(x)

        return x

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)