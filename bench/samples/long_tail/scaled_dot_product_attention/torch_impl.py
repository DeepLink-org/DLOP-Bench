# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from bench.core.executer import Executer


def scaled_dot_product_attention(q, k, v, d_key, dropout_rate):
    """
    Scaled Dot-Product Attention
    """
    product = torch.matmul(q, k).transpose(0, 1)

    weights = torch.nn.functional.softmax(product)
    if dropout_rate:
        weights = torch.nn.Dropout(weights)
    out = torch.matmul(weights, v)
    return out


def args_adaptor(np_args):
    q = torch.from_numpy(np_args[0]).cuda()
    k = torch.from_numpy(np_args[0]).cuda()
    v = torch.from_numpy(np_args[0]).cuda()
    d_key = -1
    dropout_rate = False
    return [q, k, v, d_key, dropout_rate]


def executer_creator():
    return Executer(scaled_dot_product_attention, args_adaptor)
