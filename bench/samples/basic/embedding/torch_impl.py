# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    ret = torch.nn.functional.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.long).cuda()
    weight_image = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [input_image, weight_image, np_args[2], np_args[3], np_args[4], np_args[5], np_args[6]]


def executer_creator():
    return Executer(embedding, args_adaptor)
