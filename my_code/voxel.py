import torch.nn as nn
import torch
# VoxelNet: https://arxiv.org/pdf/1711.06396.pdf


class VFE(nn.Module):
    """
    One block of voxel feature encoding (VFE),
    transforming a pointwise feature x ([max_points, in_channels])
    to [max_points, out_channels] by first passing each point through a 
    FNN. Then, aggregate all the out features through maxpooling, and concat
    the aggregated features with the original out pointwise features.
    """

    def __init__(self, in_channels, out_channels):
        super(VFE, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, max_points: int):
        '''
        x: [max_points, in_channels]
        out: [max_points, out_channels * 2]
        '''
        # assert x.size == [max_points, self.in_channels]
        pointwise_features = self.linear(x)

        aggregated_features = torch.max(pointwise_features, dim=0)[0]
        aggregated_features = aggregated_features.unsqueeze(0)

        return aggregated_features


class VoxelFeatureExtractor(nn.Module):
    """
    Voxel Feature Learning network: stacking VFE layer,
    and at the end, obtain one feature vector for each voxel
    by maxpooling all the pointwise features in the voxel.
    """

    def __init__(self, max_points, in_channels, out_channels):
        super(VoxelFeatureExtractor, self).__init__()

        self.vfe_layers = nn.ModuleList()
        for _ in range(2):
            self.vfe_layers.append(VFE(in_channels, out_channels))
            in_channels = out_channels

        self.max_points = max_points

    def forward(self, x):
        '''
        x: [max_points, in_channels]
        '''
        for i, vfe_layer in enumerate(self.vfe_layers):
            x = vfe_layer(x, self.max_points)
        # apply final maxpooling
        x = torch.max(x, dim=0)[0]
        return x


class VoxelNet(nn.Module):
    def __init__(self, max_points, in_channels, out_channels):
        super(VoxelNet, self).__init__()

        self.vfe = VoxelFeatureExtractor(max_points, in_channels, out_channels)
        self.conv3d = nn.Sequential(
            nn.Conv3d(out_channels, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channels),
        )

    def forward(self, x):
        '''
        x: [N, max_points_per_voxel, in_channels]
        N: no. of voxels
        in_channels: (x, y, z) + ancillary info such as color
        '''
        x = self.vfe(x)  # 4D tensor [C, D', H', W']
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
