import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


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
        self.conv0 = conv_blck(in_channels, 64, bn=bn)
        self.conv1 = conv_blck(self.conv0[0].out_channels, 32, bn=bn)
        self.conv2 = conv_blck(self.conv1[0].out_channels, 16, bn=bn)
        self.conv3 = nn.Conv2d(self.conv2[0].out_channels, 1, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
        return x


class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x


class CMDTop(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        chan = [128, 128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], bn=bn)
        self.conv4 = conv_blck(chan[3], chan[4], bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        return self.final(x)


class CMD60x60(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        # number of output channels
        chan = [128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], padding=2, dilation=2, bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], padding=3, dilation=3, bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], padding=4, dilation=4, bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
        return self.final(x)


class CMD120x120(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        chan = [128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], padding=4, dilation=4, bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], padding=6, dilation=6, bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], padding=8, dilation=8, bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
        return self.final(x)
        

class CMD240x240(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        chan = [128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], padding=4, dilation=4, bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], padding=12, dilation=12, bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], padding=16, dilation=16, bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
        return self.final(x)


class VGGPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=True)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = False

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x):
        outputs = []
        for layer_n in range(0, self.n_levels):
            x = self.__dict__['_modules']['level_' + str(layer_n)](x)
            outputs.append(x)

        return outputs
