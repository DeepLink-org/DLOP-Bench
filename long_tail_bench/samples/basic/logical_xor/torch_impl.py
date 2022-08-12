import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def logical_xor(logical_xor_0, logical_xor_1):
    return torch.logical_xor(logical_xor_0, logical_xor_1)

def args_adaptor(np_args):
    logical_xor_0 = torch.from_numpy(np_args[0]).cuda()
    logical_xor_1 = torch.from_numpy(np_args[1]).cuda()
    return [logical_xor_0, logical_xor_1]


def executer_creator():
    return Executer(logical_xor, args_adaptor)
