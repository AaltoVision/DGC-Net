import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_blck(in_channels, out_channels, kernel_size=3, 
              stride=1, padding=1, dilation=1, bn=False):
    if bn:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                       stride, padding, dilation),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.ReLU(inplace=True))


def conv_head(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


class CorrelationVolume(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(CorrelationVolume, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class MatchabilityNet(nn.Module):
    """ 
    Matchability network to predict a binary mask
    """
    def __init__(self, in_channels, bn=False):
        super(MatchabilityNet, self).__init__()
        self.conv0 = conv_blck(in_channels, 64, bn=batch_norm)
        self.conv1 = conv_blck(self.conv0[0].out_channels, 32, bn=batch_norm)
        self.conv2 = conv_blck(self.conv1[0].out_channels, 16, bn=batch_norm)
        self.conv3 = nn.Conv2d(self.conv2[0].out_channels, 1, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
        return x


class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):


    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        x = self.net(x)
        return x
