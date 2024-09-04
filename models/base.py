import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from pdb import set_trace


class BaseModel(nn.Module):

    def __init__(self, in_channels, out_channels=1, loss_fn=None):
        super(BaseModel, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_fn = loss_fn
            

    def get_optimizer_params(self, configs):
        configs["params"] = self.parameters()
        return configs

    def initialize(self):
        pass

    def forward(self, inputs):
        
        if isinstance(inputs, tuple):
            outputs = self._forward(*inputs)
        else:
            outputs = self._forward(inputs)
        
        if self.training:
            losses = self.losses(outputs, inputs["y"])
            return losses
        
        return outputs

    def losses(self, predictions, targets):
        loss_dict = {}
        for loss_name, loss_fn in self.loss_fn.items():
            if loss_name.startswith(("mse", "l2", "bce", "huber")):
                loss_dict[loss_name] = loss_fn(predictions, targets)
            else:
                raise ValueError("Unknown loss: {}".format(loss_name))

        return loss_dict