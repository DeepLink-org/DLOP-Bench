# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from bench.core.executer import Executer


def dec2bin(x, bits):
    mask = torch.arange(bits - 1, -1, -1, dtype=torch.int64)
    return torch.not_equal(
        x.unsqueeze(-1).bitwise_and(mask),
        torch.full([1], fill_value=0, dtype=torch.int64))


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0])
    bits = 8
    return [x, bits]


def executer_creator():
    return Executer(dec2bin, args_adaptor)
