import torch
import torch.nn as nn
import torch.nn.modules.loss as L
import torch.nn.functional as F


class L1LossMasked(L.L1Loss):
    def __init__(self):
        super(L1LossMasked, self).__init__()

    def forward(self, input, target, N_valid_pxs):
        L._assert_no_grad(target)
        return F.l1_loss(input, target, size_average=False) / (N_valid_pxs + 1e-8)