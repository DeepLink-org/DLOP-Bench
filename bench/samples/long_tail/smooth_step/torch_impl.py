# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
import torch.nn as nn
from bench.core.executer import Executer


class SmoothStep(nn.Module):
    def __init__(self, gamma=1.0):
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2.0
        self._upper_bound = gamma / 2.0
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def forward(self, inputs):
        return torch.where(
            inputs <= self._lower_bound, torch.zeros_like(inputs),
            torch.where(inputs >= self._upper_bound, torch.ones_like(inputs),
                        self._a3 * (inputs**3) + self._a1 * inputs + self._a0))


def args_adaptor(np_args):
    inputs = torch.from_numpy(np_args[0]).cuda()
    return [inputs]


def executer_creator():
    coder_instance = SmoothStep()
    return Executer(coder_instance.forward, args_adaptor)
