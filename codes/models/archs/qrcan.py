from collections import OrderedDict
from torch import nn
import torch
import numpy as np
from torch.autograd import Variable

from models.archs.q_layer import ParaCALayer
from models.archs import common
from models.archs.qhan import PCAEncoder

# from models.advanced.SAN_blocks import Nonlocal_CA
# from models.attention_manipulators.qsan_blocks import QLSRAG

class QRCAN(nn.Module):
    def __init__(self, args, pca, n_resblocks=20, n_resgroups=10, n_feats=64, in_feats=3, out_feats=3, scale=4, reduction=16,
                 res_scale=1.0, style='standard', num_metadata=10, include_pixel_attention=False,
                 selective_meta_blocks=None, num_q_layers_inner_residual=None, include_q_layer=True, **kwargs):

        super(QRCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        self.style = style

        # define head module
        modules_head = [common.default_conv(in_feats, n_feats, kernel_size)]

        # define body module
        if selective_meta_blocks is None:
            modules_body = [
                QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                               num_metadata=num_metadata, pa=include_pixel_attention, q_layer=include_q_layer,
                               act=act, res_scale=res_scale, n_resblocks=n_resblocks,
                               num_q_layers=num_q_layers_inner_residual) for _ in range(n_resgroups)]
        else:
            modules_body = []

            for index in range(n_resgroups):
                if selective_meta_blocks[index]:
                    include_q = include_q_layer
                else:
                    include_q = False
                modules_body.append(
                    QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                                   num_metadata=num_metadata, pa=include_pixel_attention, q_layer=include_q,
                                   act=act, res_scale=res_scale, n_resblocks=n_resblocks,
                                   num_q_layers=num_q_layers_inner_residual))

        self.final_body = common.default_conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, out_feats, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.encoder = PCAEncoder(torch.load(pca), cuda=True)
        

    def forward(self, x, metadata):
        metadata = self.encoder(metadata)
        x = self.head(x)
        res, *_ = self.body((x, metadata))
        res = self.final_body(res)
        res += x
        x = self.tail(res)

        return x

    def forensic(self, x, qpi, *args, **kwargs):
        x = self.head(x)
        data = OrderedDict()
        res = x
        for index, module in enumerate(self.body):
            res, res_forensic_data = module.forensic(res, qpi)
            for rcab_index, rcab_forensic_data in enumerate(res_forensic_data):
                data['R%d.C%d' % (index, rcab_index)] = rcab_forensic_data
        res = self.final_body(res)
        res += x
        x = self.tail(res)
        return x, data

class QRCAB(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(
            self, conv, n_feat, kernel_size, reduction, style='standard', pa=False, q_layer=False,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, num_metadata=1):

        super(QRCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        self.final_body = QCALayer(channel=n_feat, reduction=reduction, style=style, num_metadata=num_metadata)
        self.pa = pa
        self.q_layer = q_layer
        if pa:
            self.pa_node = PALayer(channel=n_feat)
        if q_layer:
            self.q_node = ParaCALayer(network_channels=n_feat, num_metadata=num_metadata, nonlinearity=True)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x[0])
        res = self.final_body(res, x[1])
        if self.pa:
            res = self.pa_node(res)
        if self.q_layer:
            res = self.q_node(res, x[1])
        res += x[0]
        return res, x[1]

    def forensic(self, x, qpi):

        res = self.body(x)
        conv_data = []
        for module in self.body:
            if isinstance(module, nn.Conv2d):
                conv_data.append(module.weight.detach().cpu().numpy().flatten())

        res, forensic_data = self.final_body.forensic(res, qpi)
        if self.pa:
            res, forensic_pa = self.pa_node.forensic(res)
            forensic_data['pixel_attention_map'] = forensic_pa
        if self.q_layer:
            res, meta_attention = self.q_node.forensic(res, qpi)
            forensic_data['meta_attention_map'] = meta_attention

        forensic_data['pre-residual'] = res
        forensic_data['pre-residual-flat'] = res.cpu().numpy().flatten()
        res += x
        forensic_data['post-residual'] = res
        forensic_data['post-residual-flat'] = res.cpu().numpy().flatten()
        forensic_data['conv_flat'] = np.hstack(np.array(conv_data))
        return res, forensic_data


# Residual Group (RG)
class QResidualGroup(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, style, num_metadata,
                 pa, q_layer, num_q_layers):
        super(QResidualGroup, self).__init__()
        modules_body = []

        for index in range(n_resblocks):
            if num_q_layers is None or index < num_q_layers:
                q_in = q_layer
            else:
                q_in = False
            modules_body.append(QRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                                      act=act, res_scale=res_scale, style=style,
                                      pa=pa, q_layer=q_in, num_metadata=num_metadata))

        self.final_body = conv(n_feat, n_feat, kernel_size)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res, _ = self.body(x)
        res = self.final_body(res)
        res += x[0]
        return res, x[1]

    def forensic(self, x, qpi):
        res = x
        forensic_data = []
        for module in self.body:
            res, RCAB_data = module.forensic(res, qpi)
            forensic_data.append(RCAB_data)
        res = self.final_body(res)
        res += x
        return res, forensic_data
    

class QCALayer(nn.Module):
    """
    Combined channel-attention and meta-attention layer.  Diverse style choices available.
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, channel, style, reduction=16, num_metadata=1):
        """
        :param channel:  Network feature map channel count.
        :param style: Type of attention to use.  Options are:
        modulate:  Normal channel attention occurs, but meta-vector is multiplied with the final attention
        vector prior to network modulation.
        mini_concat:  Concatenate meta-vector with inner channel attention vector.
        max_concat:  Concatenate meta-vector with feature map aggregate, straight after average pooling.
        softmax:  Implements max_concat, but also applies softmax after the final FC layer.
        extended_attention: Splits attention into four layers, and adds metadata vector in second layer.
        standard:  Do not introduce any metadata.
        :param reduction: Level of downscaling to use for inner channel attention vector.
        :param num_metadata: Expected metadata input size.
        """
        super(QCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight

        if reduction < 16:
            raise RuntimeError('Using an extreme channel attention reduction value')

        if style == 'modulate' or style == 'mini_concat' or style == 'standard':
            channel_in = channel
        else:
            channel_in = channel + num_metadata

        channel_reduction = channel // reduction

        if style == 'modulate' or style == 'max_concat' or style == 'softmax' or style == 'standard':
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel_in, channel_reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif style == 'mini_concat':
            self.pre_concat = nn.Conv2d(channel_in, channel_reduction, 1, padding=0, bias=True)
            self.conv_du = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_reduction + num_metadata, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif style == 'extended_attention':

            channel_fractions = [(channel_in, channel//2),
                                 (channel//2 + num_metadata, channel//4),
                                 (channel//4 + num_metadata, channel_reduction)]
            self.feature_convs = nn.ModuleList()
            for (inp, outp) in channel_fractions:
                self.feature_convs.append(
                    nn.Sequential(
                        nn.Conv2d(inp, outp, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True)
                    )
                )
            self.final_conv = nn.Sequential(
                nn.Conv2d(channel_reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        if style == 'softmax':
            self.softmax = nn.Softmax(dim=1)

        self.style = style

    def forward(self, x, attributes):

        y = self.avg_pool(x)
        if self.style == 'modulate':
            y = self.conv_du(y) * attributes
        elif self.style == 'max_concat':
            y = self.conv_du(torch.cat((y, attributes), dim=1))
        elif self.style == 'mini_concat':
            y = self.pre_concat(y)
            y = self.conv_du(torch.cat((y, attributes), dim=1))
        elif self.style == 'extended_attention':
            for conv_section in self.feature_convs:
                y = conv_section(torch.cat((y, attributes), dim=1))
            y = self.final_conv(y)
        elif self.style == 'softmax':
            y = self.conv_du(torch.cat((y, attributes), dim=1))
            y = self.softmax(y)
        elif self.style == 'standard':
            y = self.conv_du(y)
        else:
            raise NotImplementedError

        return x * y
