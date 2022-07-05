import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def find_first_diff(input_torch):
    for i in range(len(input_torch[0].shape)):
        cur = input_torch[0].shape[i]
        for j in range(len(input_torch)):
            if input_torch[j].shape[i] != cur:
                return i
    return 0

def cat(tensor_input):
    input_torch = []
    dim = 0
    for i in range(len(tensor_input[0])):
        input_np = np.random.random(tensor_input[0][i])
        input_torch.append(torch.from_numpy(input_np).to(torch.float32).cuda())
    if len(input_torch[0].shape) > 1 and len(input_torch) > 1:
        dim = find_first_diff(input_torch)
    return torch.cat(input_torch, dim)


def args_adaptor(np_args):
    input_tensor = np_args
    return [input_tensor]


def executer_creator():
    return Executer(cat, args_adaptor)
