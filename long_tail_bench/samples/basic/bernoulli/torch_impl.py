import torch
import numpy as np
from long_tail_bench.core.executer import Executer


def bernoulli(input_tensor):
    ret = torch.bernoulli(input_tensor)
    return ret

def args_adaptor(np_args):
    np_input = np_args[0]
    input_tensor = torch.from_numpy(np_input).to(torch.float32).cuda()
    return [input_tensor]

def executer_creator():
    return Executer(bernoulli, args_adaptor)