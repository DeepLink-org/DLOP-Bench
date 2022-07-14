from audioop import bias
import torch
from long_tail_bench.core.executer import Executer


def group_norm(input, num_groups, weight, bias, eps):
    output = torch.group_norm(input, num_groups, weight, bias, eps)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    weight = torch.from_numpy(np_args[2]).cuda()
    bias = torch.from_numpy(np_args[3]).cuda()
    return [input, np_args[1], weight, bias, np_args[4]]


def executer_creator():
    return Executer(group_norm, args_adaptor)
