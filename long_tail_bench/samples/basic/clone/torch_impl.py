import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def clone(clone_0):
    return torch.clone(clone_0)

def args_adaptor(np_args):
    clone_0 = torch.from_numpy(np_args[0])
    return [clone_0]


def executer_creator():
    return Executer(clone, args_adaptor)
