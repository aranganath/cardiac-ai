import torch

from torch import nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss, BCEWithLogitsLoss, HuberLoss
from pdb import set_trace


class CustomMSELoss(MSELoss):

    def __init__(self, weight=None, reduction="mean"):
        super(CustomMSELoss, self).__init__(reduction=reduction)
        self.weight = weight

    @torch.enable_grad()
    def forward(self, input, target):
        loss = super(CustomMSELoss, self).forward(input, target)
        if self.weight is not None:
            loss = self.weight * loss
        return loss


class L2Loss(nn.Module):

    def __init__(self, weight=None):
        super(L2Loss, self).__init__()
        self.weight = weight
    
    @torch.enable_grad()
    def forward(self, input, target):
        # TODO(june): This is just a L2 loss. Make it in general if necessary.
        loss = torch.sqrt(torch.sum(torch.pow((input - target), 2)))
        if self.weight is not None:
            loss = self.weight * loss
        return loss
    


class BCELoss(BCEWithLogitsLoss):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.weight = weight

    @torch.enable_grad()
    def forward(self, input, target):
        loss = super(BCELoss, self).forward(input, target)
        if self.weight is not None:
            loss = self.weight * loss

        return loss



class HuberLoss(HuberLoss):
    def __init__(self, weight=None):
        super(HuberLoss, self).__init__()
        self.weight = weight

    @torch.enable_grad()
    def forward(self, input, target):
        loss = super(HuberLoss, self).forward(input, target)
        if self.weight is not None:
            loss = self.weight * loss

        return loss

