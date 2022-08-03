from functools import reduce
import torch
from torch.nn import functional
from long_tail_bench.core.executer import Executer


def instance_norm(input, running_mean, running_var,
            weight, bias, use_input_stats, momentum, eps):
    output = functional.instance_norm(input, running_mean, running_var,
            weight, bias, use_input_stats, momentum, eps)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
 
    return [input, np_args[1], np_args[2], np_args[3], np_args[4], np_args[5], np_args[6], np_args[7]]


def executer_creator():
    return Executer(instance_norm, args_adaptor)
