import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def expand_as(tensor1, tensor2):
    return tensor1.expand_as(tensor2)


def args_adaptor(np_args):
    tensor1 = torch.from_numpy(np_args[0]).cuda()
    tensor2 = torch.from_numpy(np_args[1]).cuda()
    return [tensor1, tensor2]


def executer_creator():
    return Executer(expand_as, args_adaptor)
