#########################################################
# Ref: https://github.com/hqqxyy/VFIPS
#########################################################
import torch
import torch.nn as nn
from .common import normalize_tensor
from .swinir import SwinDiffTiny


class Extractor(torch.nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleFirst = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSecond = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThird = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFourth = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFifth = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSixth = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )


    def forward(self, tensorInput):
        tensorFirst = self.moduleFirst(tensorInput)
        tensorSecond = self.moduleSecond(tensorFirst)
        tensorThird = self.moduleThird(tensorSecond)
        tensorFourth = self.moduleFourth(tensorThird)
        tensorFifth = self.moduleFifth(tensorFourth)
        tensorSixth = self.moduleSixth(tensorFifth)

        return [tensorFirst, tensorSecond, tensorThird, tensorFourth, tensorFifth, tensorSixth]


class MultiScaleV33(torch.nn.Module):
    def __init__(self):
        super(MultiScaleV33, self).__init__()
        self.moduleExtractor = Extractor()
        chns = [16, 32, 64, 96, 128]
        out_chn = 32
        merge_layers = []
        for chn in chns:
            merge_layers.append(
                SwinDiffTiny(in_chans=12 * chn * 3, out_chans=out_chn, embed_dim=32, depths=[1,], num_heads=[2,], window_size=4, mlp_ratio=2.,)
            )
        self.merge_layers = nn.ModuleList(merge_layers)

    def forward(self, inputFirst, inputSecond):
        B, V, C, H, W = inputFirst.size()
        inputFirst = inputFirst.view(B * V, C, H, W)
        inputSecond = inputSecond.view(B * V, C, H, W)

        # We skip the last features, due to the swin transformer
        tensorFeasFirst = self.moduleExtractor(inputFirst)[: -1]
        tensorFeasSecond = self.moduleExtractor(inputSecond)[: -1]

        tensorFeas = []
        for tensorFeaFirst, tensorFeaSecond in zip(tensorFeasFirst, tensorFeasSecond):
            tensorFeaFirst = normalize_tensor(tensorFeaFirst)
            tensorFeaSecond = normalize_tensor(tensorFeaSecond)
            tensorFea = torch.abs(tensorFeaFirst - tensorFeaSecond)
            tensorFea = torch.cat([tensorFeaFirst, tensorFeaSecond, tensorFea], dim=1)
            tensorFeas.append(tensorFea)

        tensorFeas = [p.view(B, -1, p.size(2), p.size(3)) for p in tensorFeas]
        tensorOuts = []
        for merge_layer, tensorFea in zip(self.merge_layers, tensorFeas):
            tensorOut = merge_layer(tensorFea)
            tensorOuts.append(tensorOut)
        return tensorOuts

