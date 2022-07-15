import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def exp(exp_0):
    return torch.exp(exp_0)

def args_adaptor(np_args):
    exp_0 = torch.from_numpy(np_args[0])
    return [exp_0]


def executer_creator():
    return Executer(exp, args_adaptor)
