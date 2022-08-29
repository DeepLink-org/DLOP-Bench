# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch
import torch.nn.functional as F
from bench.core.executer import Executer


def channel_attention(attention_mat, attention, *channel_embeddings):
    """
        channel_embeddings_1: (num_user, emb_size)
        attention_mat: (emb_size, emb_size)
        attention: (1, emb_size)
    """
    weights = []
    attention_mat_weight = attention_mat
    attention_weight = attention
    for embedding in channel_embeddings:
        # ((num_user, emb_size) * (emb_size, emb_size)) @ (1, emb_size) = (num_user, emb_size) @ (1, emb_size)  # noqa
        # = (num_user, emb_size) -> (num_user, )
        weights.append(torch.sum(torch.multiply(
            torch.matmul(
                embedding, attention_mat_weight), attention_weight), 1))
    t = torch.stack(weights)
    # (num_user, channel_num)
    score = F.softmax(torch.transpose(t, 1, 0))
    mixed_embeddings = 0.0
    for i in range(len(weights)):
        # (emb_size, num_user) @
        # (num_user, emb_size) @ (num_user, 1) -> (num_user, emb_size)
        mixed_embeddings += torch.transpose(
            torch.multiply(torch.transpose(channel_embeddings[i], 1, 0),
                           torch.transpose(score, 1, 0)[i]), 1, 0)
    return mixed_embeddings, score


def args_adaptor(np_args):
    attention_mat = torch.from_numpy(np_args[0]).cuda()
    attention = torch.from_numpy(np_args[1]).cuda()
    channel_embeddings = torch.from_numpy(np_args[2]).cuda()
    return [attention_mat, attention, channel_embeddings]


def executer_creator():
    return Executer(channel_attention, args_adaptor)
