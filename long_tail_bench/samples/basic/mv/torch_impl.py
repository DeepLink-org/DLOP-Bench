from functools import reduce
import torch
from long_tail_bench.core.executer import Executer


def mv(input, vec):
    output = torch.mv(input, vec)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    vec = torch.from_numpy(np_args[1]).cuda()
    
    return [input, vec]


def executer_creator():
    return Executer(mv, args_adaptor)