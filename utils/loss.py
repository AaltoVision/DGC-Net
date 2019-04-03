import torch
import torch.nn as nn
import torch.nn.modules.loss as L
import torch.nn.functional as F


class L1LossMasked(L.L1Loss):
    def __init__(self):
        super(L1LossMasked, self).__init__()

    def forward(self, est, target, N_valid_pxs):
        """
        Args:
            est: grid (correspondence) estimates [BxHxWx2]
            target: target values [BxHxWx2]
            N_valid_pxs: number of valid correspondences [value]
        Output:
            L1 loss [value]
        """
        return F.l1_loss(est,
                         target,
                         size_average=False) / (N_valid_pxs + 1e-8)
