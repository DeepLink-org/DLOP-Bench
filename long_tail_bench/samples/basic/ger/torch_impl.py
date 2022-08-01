import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def ger(input, vec2):
    ret = torch.ger(input, vec2)
    return ret

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    vec2 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [input, vec2]


def executer_creator():
    return Executer(ger, args_adaptor)
