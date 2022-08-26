# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def margin_ranking_loss(input1_torch, input2_torch, target_torch, margin, reduction):
    return torch.nn.functional.margin_ranking_loss(input1_torch, input2_torch,
            target_torch, margin=margin, reduction=reduction)


def args_adaptor(np_args):
    margin = np_args[3]
    reduction = np_args[4]
    input1_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input2_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    target_torch = torch.from_numpy(np_args[2]).to(torch.float32).cuda()

    return [input1_torch, input2_torch, target_torch, margin, reduction]


def executer_creator():
    return Executer(margin_ranking_loss, args_adaptor)
