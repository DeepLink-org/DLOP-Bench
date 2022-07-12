import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def stackBP(input_torch, axis):
    ret = torch.stack(input_torch, axis)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_np = np_args[0]
    input_torch = []
    for i in range(len(np_args[0])):
        input_torch.append(torch.tensor(input_np[i], requires_grad=True).to(torch.float32).cuda())
    axis = np_args[1]
    
    return [input_torch, axis]


def executer_creator():
    return Executer(stackBP, args_adaptor)
