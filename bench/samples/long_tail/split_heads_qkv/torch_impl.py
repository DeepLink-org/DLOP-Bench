# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch
from bench.core.executer import Executer


def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
    """
    Reshape input tensors at the last dimension to split multi-heads
    and then transpose. Specifically, transform the input tensor with shape
    [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
    with shape [bs, n_head, max_sequence_length, hidden_dim].
    """
    # The value 0 in shape attr means copying the corresponding dimension
    # size of the input as the output dimension size.
    reshaped_q = torch.reshape(queries, shape=[1, 1, n_head, d_key])
    # permuate the dimensions into:
    # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
    q = torch.transpose(reshaped_q, 1, 2)
    # For encoder-decoder attention in inference, insert the ops and vars
    # into global block to use as cache among beam search.
    reshaped_k = torch.reshape(keys, [1, 1, n_head, d_key])
    k = torch.transpose(reshaped_k, 1, 2)
    reshaped_v = torch.reshape(values, shape=[1, 1, n_head, d_value])
    v = torch.transpose(reshaped_v, 1, 2)
    return q, k, v


def args_adaptor(np_args):
    queries = torch.from_numpy(np_args[0]).cuda()
    keys = torch.from_numpy(np_args[1]).cuda()
    values = torch.from_numpy(np_args[2]).cuda()
    n_head = 12
    d_key = -1
    d_value = -1
    return [queries, keys, values, n_head, d_key, d_value]


def executer_creator():
    return Executer(__split_heads_qkv, args_adaptor)
