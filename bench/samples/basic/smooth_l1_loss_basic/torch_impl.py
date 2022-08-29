# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
from torch.nn import functional
from bench.core.executer import Executer


def smooth_l1_loss_basic(x1, x2):
    output = functional.smooth_l1_loss(x1, x2)
    return output


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x2 = torch.from_numpy(np_args[1]).cuda()
    return [x1, x2]


def executer_creator():
    return Executer(smooth_l1_loss_basic, args_adaptor)
