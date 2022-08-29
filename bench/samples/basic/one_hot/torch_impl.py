import torch
from torch.nn import functional
from long_tail_bench.core.executer import Executer


def one_hot(input, num_classes):
    output = functional.one_hot(input, num_classes)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()

    return [input, np_args[1]]


def executer_creator():
    return Executer(one_hot, args_adaptor)
