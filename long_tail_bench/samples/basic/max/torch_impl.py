import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def max(input_torch, dim):
    return torch.max(input_torch, dim)

def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch, np_args[1][0]]


def executer_creator():
    return Executer(max, args_adaptor)
