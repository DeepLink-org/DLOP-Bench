import torch
import torch.nn
from long_tail_bench.core.executer import Executer


def relu(input_torch, inplace):
    return torch.nn.functional.relu(input_torch, inplace=inplace)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).cuda()
    return [input_torch, np_args[1]]


def executer_creator():
    return Executer(relu, args_adaptor)
