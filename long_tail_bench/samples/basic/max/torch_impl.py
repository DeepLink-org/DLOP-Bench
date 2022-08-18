import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def max(max1, dim):
    if dim[0] == -999:
        return torch.max(max1)
    return torch.max(max1, dim[0])

def args_adaptor(np_args):
    max1 = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [max1, np_args[1]]


def executer_creator():
    return Executer(max, args_adaptor)
