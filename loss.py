import torch
import torch.nn as nn

class RealBCELoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(RealBCELoss, self).__init__()
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        logit = torch.nn.functional.sigmoid(d0 - d1)
        if len(logit.size()) == 4:
            assert logit.size(2) == 1
            assert logit.size(3) == 1
            logit = logit[:, :, 0, 0]
        return self.loss(logit, judge)


class RealL1Loss(nn.Module):
    def __init__(self, reduction='sum'):
        super(RealL1Loss, self).__init__()
        self.loss = torch.nn.L1Loss(reduction=reduction)

    def forward(self, d0, d1, judge):
        logit = torch.nn.functional.sigmoid(d0 - d1)
        if len(logit.size()) == 4:
            assert logit.size(2) == 1
            assert logit.size(3) == 1
            logit = logit[:, :, 0, 0]
        return self.loss(logit, judge)

