import torch
import torch.nn.functional as F
from bench.core.executer import Executer


def label_aware_attention(keys, query):
    """label_aware_attention
    """
    weight = torch.sum(keys * query, axis=-1, keepdim=True)
    weight = torch.pow(weight, 2)  # [x,k_max,1]
    weight = F.softmax(weight, 1)
    output = torch.sum(keys * weight, axis=1)
    return output, weight


def args_adaptor(np_args):
    keys = torch.from_numpy(np_args[0]).cuda()
    query = torch.from_numpy(np_args[1]).cuda()
    return [keys, query]


def executer_creator():
    return Executer(label_aware_attention, args_adaptor)
