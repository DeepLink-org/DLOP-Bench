import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def diagflat(input):
    ret = torch.diagflat(input)
    return ret

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input]


def executer_creator():
    return Executer(diagflat, args_adaptor)
