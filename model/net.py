import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict

from model.modules import *


def factory(type, in_channels):
    if type == 'level_0':
        return CMD240x240(in_channels=in_channels, bn=True)
    elif type == 'level_1':
        return CMD120x120(in_channels=in_channels, bn=True)
    elif type == 'level_2':
        return CMD60x60(in_channels=in_channels, bn=True)
    elif type == 'level_3':
        return CMDTop(in_channels=in_channels, bn=True)
    elif type == 'level_4':
        return CMDTop(in_channels=in_channels, bn=True)
    assert 0, 'Correspondence Map Decoder bad creation: ' + type



class DGCNet(nn.Module):
    def __init__(self, mask=False)
        super(DGCNet, self).__init__()

        self.mask = mask

        self.pyramid = VGGPyramid()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()
        # Correlation volume
        self.corr = CorrelationVolume()

        if self.mask:
            self.matchability_net = MatchabilityNet(in_channels=128, bn=True)

        # create a hierarchy of correspondence decoders
        map_dim = 2
        N_out = [x + map_dim for x in [128, 128, 256, 512, 225]]

        for i, in_chan in enumerate(N_out):
            self.__dict__['_modules']['reg_' + str(i)] = factory('level_' + str(i), in_chan) 


    def forward(self, x1, x2):

