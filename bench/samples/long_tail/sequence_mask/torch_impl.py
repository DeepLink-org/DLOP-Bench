# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from bench.core.executer import Executer


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len,
                         device=lengths.device).type_as(lengths).repeat(
                             batch_size, 1).lt(lengths.unsqueeze(1)))


def args_adaptor(np_args):
    input0 = torch.from_numpy(np_args[0]).cuda()
    return [input0]


def executer_creator():
    return Executer(sequence_mask, args_adaptor)
