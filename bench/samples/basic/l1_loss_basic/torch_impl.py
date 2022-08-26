# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def l1_loss(input_torch, target_torch, reduction):
    ret = torch.nn.functional.l1_loss(input_torch, target_torch, reduction=reduction)
    return ret


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    target_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()

    return [input_torch, target_torch, np_args[2]]


def executer_creator():
    return Executer(l1_loss, args_adaptor)
