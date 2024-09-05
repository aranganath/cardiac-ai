#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019

from torch import optim


class StepLR(optim.lr_scheduler.StepLR):
    def __init__(self, **args):
        self.by_epoch = False
        super(StepLR, self).__init__(**args)


class MultiStepLR(optim.lr_scheduler.MultiStepLR):
    def __init__(self, **args):
        self.by_epoch = True
        super(MultiStepLR, self).__init__(**args)


class PolynomialLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        self.by_epoch = False # by step
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]


class CosineAnnealing(optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, **args):
        self.by_epoch = False
        super(CosineAnnealing, self).__init__(**args)
