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
        # Bilinear upsampler
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear')

        if self.mask:
            self.matchability_net = MatchabilityNet(in_channels=128, bn=True)

        # create a hierarchy of correspondence decoders
        map_dim = 2
        N_out = [x + map_dim for x in [128, 128, 256, 512, 225]]

        for i, in_chan in enumerate(N_out):
            self.__dict__['_modules']['reg_' + str(i)] = factory('level_' + str(i), in_chan) 


    def forward(self, x1, x2):
        """
        x1 - target image
        x2 - source image
        """

        target_pyr = self.pyramid(x1)
        source_pyr = self.pyramid(x2)

        # do feature normalisation
        feat_top_pyr_trg = self.l2norm(target_pyr[-1])
        feat_top_pyr_src = self.l2norm(source_pyr[-1])

        # do correlation
        corr1 = self.corr(feat_top_pyr_trg, feat_top_pyr_src)
        corr1 = self.l2norm(F.relu(corr1))

        b, c, h, w = corr1.size()
        init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        est_grid = self.__dict__['_modules']['reg_4'](x1=corr1, x3=init_map)

        estimates_grid = [est_grid]

        for k in reversed(range(4)):
            p1, p2 = target_pyr[k], source_pyr[k]
            est_map = self.upsampler(esimates_grid[-1])

            p1_w = F.grid_sample(p1, est_map.transpose(1,2).transpose(2,3))
            est_map = self.__dict__['_modules']['reg_' + str(k)](x1=p1_w, x2=p2, x3=est_map)
            estimates_grid.append(est_map)

        '''
        p1, p2 = target_pyr[3], source_pyr[3]
        est_map = self.upsampler(estimates_grid[-1])

        p1_w = F.grid_sample(p1, est_flow.transpose(1,2).transpose(2,3))
        
        # regressing the correspondence map
        est_map = self.__dict__['_modules']['reg_3'](x1=p1_w, x2=p2, x3=est_map)
        estimates_grid.append(est_map)

        p1, p2 = target_pyr[2], source_pyr[2]
        est_map = self.upsampler(estimates_grid[-1])

        p1_w = F.grid_sample(p1, est_map.transpose(1,2).transpose(2,3))

        # regressing the correspondence map
        est_map = self.__dict__['_modules']['reg_2'](x1=p1_w, x2=p2, x3=est_map)
        estimates_grid.append(est_map)

        p1, p2 = target_pyr[1], source_pyr[1]
        est_map = self.upsampler(estimates_grid[-1])

        p1_w = F.grid_sample(p1, est_map.tranpose(1,2).transpose(2,3))

        # regressing the correspondence map
        est_map = self.__dict__['_modules']['reg_1'](x1=p1_w, x2=p2, x3=est_map)
        estimates_grid.append(est_map)

        p1, p2 = target_pyr[0], source_pyr[0]
        est_map = self.upsampler(estimates_grid[-1])

        p1_w = F.grid_sample(p1, est_map.transpose(1,2).transpose(2,3))

        est_map = self.__dict__['_modules']['reg_0'](x1=p1_w, x2=p2, init_flow=est_flow)
        estimates_grid.append(est_flow)
        '''

        matchability = None
        if self.mask:
            matchability = self.matchability_net(x1=p1_w, x2=p2)

        return estimates_grid, matchability

