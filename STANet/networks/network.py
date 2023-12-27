import torch
import torch.nn as nn
import torch.nn.functional as F


# Pyramid Network for Feature Extraction
class PyramidNetwork(nn.Module):
    def __init__(self, num_levels=6, initial_channels=16):
        super(PyramidNetwork, self).__init__()

        self.features = nn.ModuleList()
        for i in range(num_levels):
            self.features.append(self._level_block(initial_channels * (2**i)))

    def _level_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feature_maps = []
        for feature in self.features:
            x = feature(x)
            feature_maps.append(x)
        return feature_maps


class STANet(nn.Module):
    def __init__(self):
        super(STANet, self).__init__()

        # Two 3D convolution blocks
        self.conv3d_block1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.conv3d_block2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, patch_quality_indices, combined_features):
        # Resample combined features and patch quality indices to the desired shape [WP, HP, TP]
        reorganized_tensor = torch.stack(combined_features).reshape(10, 9, 16, 12, 256, 4, 4)
        input_tensor = reorganized_tensor.permute(0, 4, 1, 2, 3, 5, 6)
        combined_features_reshaped = input_tensor.reshape(10, 256, 16, 9, 4*4*12)
        # combined_features_reshaped = F.interpolate(combined_features, size=(10, 9, 16), mode='trilinear')
        # Reshape the tensor
        reshaped_tensor = patch_quality_indices.view(1, 1, 10, 9, 16)

        # Interpolate
        patch_quality_indices_reshaped = F.interpolate(reshaped_tensor, size=(10, 9, 16), mode='trilinear')
        #*** patch_quality_indices_reshaped = F.interpolate(patch_quality_indices, size=(10, 9, 16), mode='trilinear')

        # Pass through 3D convolution blocks
        x = self.conv3d_block1(combined_features_reshaped)
        x = self.conv3d_block2(x)
        x = torch.mean(x, dim=-1)
        # Apply Softmax to get weights along the channel dimension
        weights = F.softmax(x, dim=1)

        # Weight the patch quality indices
        weighted_quality_indices = weights.squeeze(1) * patch_quality_indices_reshaped.squeeze().permute(0, 2, 1)

        # Combine weighted indices to produce the final quality score
        final_quality_score = torch.mean(weighted_quality_indices)

        return final_quality_score.unsqueeze(0)
