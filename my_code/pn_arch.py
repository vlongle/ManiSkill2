'''
File: /pn_arch.py
Project: my_code
Created Date: Thursday June 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

"""
 Less efficient implementation of pointnet
 that doesn't use conv1d but easier to understand. Based 
 on: 
 https://github.com/Kami-code/dexart-release/blob/main/stable_baselines3/networks/pretrain_nets.py
We should use the more efficient implementation in pn_arch_eff.py (WIP) in future.
"""


# feat_dim -> 64 -> 128 -> 1024
# FC block: Linear -> LayerNorm1D -> ReLU.




import torch
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        # TODO: check if sb3 handles LayerNorm correctly (e.g., turning on model.test
        # and model.train appropriately)
        # self.ln = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        # x = self.ln(x)
        x = self.relu(x)
        return x


class PointNet(nn.Module):
    def __init__(self, feat_dim=3):
        super(PointNet, self).__init__()

        #  [64, 128, 512] from https://arxiv.org/pdf/2302.04659.pdf
        self.local_mlp = nn.Sequential(
            Block(feat_dim, 64),
            Block(64, 128),
            Block(128, 512)
            # Block(64, 256),
        )

    def forward(self, x):
        """
        X: [B, N, 3]
        """
        x = self.local_mlp(x)  # -> [B, N, out_dim]
        x = torch.max(x, dim=1)[0]  # -> [B, out_dim]
        return x


if __name__ == "__main__":
    batchsize = 32
    num_points = 100
    pointcloud = torch.rand(batchsize, num_points, 3)
    feat_extractor = PointNet()
    feats = feat_extractor(pointcloud)
    print("feats.shape", feats.shape)  # (batchsize, feat_dim=1024)
