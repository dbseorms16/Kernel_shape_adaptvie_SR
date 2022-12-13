from models.archs import common
import torch
import torch.nn as nn
import pdb

def make_model(args, parent=False):
    return HAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
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

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        # self.kawm1 = KAWM(n_feat)
        # self.kawm2 = KAWM(n_feat)
        
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
            # if name == '0' and type(midlayer).__name__ == 'Conv2d':
            #     res = self.kawm1(res, x[1])
            # if name == '2' and type(midlayer).__name__ == 'Conv2d':
            #     res = self.kawm2(res, x[1])
                
        #res = self.body(x).mul(self.res_scale)
        res += x[0]
        return res, x[1]


class KAWM(nn.Module):
      
    def __init__(self, in_channel, kernel_size=3):

        super(KAWM, self).__init__()
        
        self.transformer = nn.Conv2d(in_channel, in_channel, (3, 1), bias=False,
                                     padding=(1, 0), groups=in_channel, padding_mode='replicate')
        
        self.kernel_transformer = nn.Sequential(
            nn.Conv2d(1, in_channel*2, 2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel*2, in_channel, 2, padding=0, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        
        constant_init(self.transformer, val=0)

        # self.transformer_v = nn.Conv2d(in_channel, in_channel, (1, 3), bias=False,
        #                              padding=(0, 1), groups=in_channel, padding_mode='replicate')
        
        # self.kernel_transformer_v = nn.Sequential(
        #     nn.Conv2d(1, in_channel*2, 2, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channel*2, in_channel, 2, padding=0, bias=True),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid()
        # )        
        # constant_init(self.transformer_v, val=0)
        
    def forward(self, x, kernel):
        b, c,_,_ = x.size()
        # dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # x = torch.rot90(x, 1, [2,3])
        
        
        kernel = kernel.reshape(b, 1, 21, 21)
        # kernel = torch.rot90(kernel, -1, [2,3])
        y = self.kernel_transformer(kernel) 
        moduled_x = self.transformer(x) * y 
        
        # y = self.kernel_transformer_v(kernel) 
        # moduled_y = self.transformer_v(x) * y 
        
        # x = x + moduled_x + moduled_y
        # x = x + moduled_x 
        # x = x 
        # x = x + moduled_x + moduled_y 

        # x = torch.rot90(x, -1, [2,3])
        
        return x + moduled_x

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
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

from torch.autograd import Variable
class PCAEncoder(object):
    def __init__(self, weight, cuda=False):
        self.weight = weight #[l^2, k]
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, C, H, W = batch_kernel.size() #[B, l, l]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))

## Holistic Attention Network (HAN)
class HAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HAN, self).__init__()
        
        n_resgroups = args['n_resgroups']
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        reduction = args['reduction'] 
        scale = args['scale']
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args['rgb_range'], rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args['n_colors'], n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args['res_scale'], n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args['n_colors'], kernel_size)]

        self.add_mean = common.MeanShift(args['rgb_range'], rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)
        

    def forward(self, x, kernel):
        # x = self.sub_mean(x)
        b, c, w, h = x.size()
        kernel = kernel.reshape(b, 1, 21, 21)
        
        x = self.head(x)
        res = x
        #pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            if type(midlayer).__name__ == 'ResidualGroup':
                res, _ = midlayer((res, kernel))
            else:
                res = midlayer(res)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        
        res += x
        #res = self.csa(res)

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') >= 0:
    #                     print('Replace pre-trained upsampler to new one...')
    #                 else:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))

    #     if strict:
    #         missing = set(own_state.keys()) - set(state_dict.keys())
    #         if len(missing) > 0:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(missing))
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)