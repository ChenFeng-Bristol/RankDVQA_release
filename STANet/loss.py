import torch
import torch.nn as nn

# Loss function
class STANetLoss(nn.Module):
    def __init__(self):
        super(STANetLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output1, output2, mos1, mos2):
        diff_output = output1 - output2
        diff_mos = mos1 - mos2
        return self.mse_loss(diff_output, diff_mos)