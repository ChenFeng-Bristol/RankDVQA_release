#########################################################
# Ref: https://github.com/hqqxyy/VFIPS
#########################################################
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from .common import *
from .multi_scale import *

class LPIPS_3D_Diff(nn.Module):
    def __init__(self,
                 net='alex_3d',
                 lpips_method='lpips',
                 dksize=1,
                 ckpt_path='',
                 opt=None,
                 ):

        super(LPIPS_3D_Diff, self).__init__()

        self.pnet_type = net
        self.spatial = False
        self.lpips_method = lpips_method
        self.scaling_layer = ScalingLayer()


        if (self.pnet_type == 'multiscale_v33'):
            net_type = MultiScaleV33
            self.net = net_type()
            self.chns = [32, 32, 32, 32, 32]
        else:
            raise NotImplementedError

        self.L = len(self.chns)

        if self.lpips_method == 'lpips':
            use_dropout = True
            lins = []
            for chn in self.chns:
                lins.append(NetLinLayer(chn, use_dropout=use_dropout))
            self.lins = nn.ModuleList(lins)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        if ckpt_path != '':
            print('=> load pretrained model', ckpt_path)
            self.net.moduleExtractor.load_state_dict(torch.load(ckpt_path))


    def forward(self, in0, in1, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        diffs = self.net.forward(in0_input, in1_input)

        if self.lpips_method == 'lpips':
            if (self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            val = res[0]
            for l in range(1, self.L):
                val += res[l]
        elif self.lpips_method == 'at1':
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            res = torch.cat(res, dim=1)
            val = self.lins2(res)
        else:
            if (self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

            val = res[0]
            for l in range(1, self.L):
                val += res[l]

        return val

#