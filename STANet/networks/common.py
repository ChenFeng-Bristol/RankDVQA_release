import torch
import torch.nn as nn

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        device = inp.device  # Get the device from the input tensor
        self.shift = self.shift.to(device)
        self.scale = self.scale.to(device)
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


def cvt_5d_4d(tensor, B, V):
    _, _, C, H, W = tensor.size()  # update CHW
    return tensor.view(B, V * C, H, W)


def cvt_4d_5d(tensor, B, V):
    N, C, H, W = tensor.size()  # update CHW
    assert N == B * V
    return tensor.view(B, V, C, H, W)


