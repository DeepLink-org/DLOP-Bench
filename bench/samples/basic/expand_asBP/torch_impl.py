import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def expand_as(tensor1, tensor2):
    tensor1.requires_grad=True
    tensor1 = tensor1.expand_as(tensor2)
    tensor1.backward(tensor1)
    return tensor1

def args_adaptor(np_args):
    tensor1 = torch.from_numpy(np_args[0]).cuda()
    tensor2 = torch.from_numpy(np_args[1]).cuda()
    return [tensor1, tensor2]


def executer_creator():
    return Executer(expand_as, args_adaptor)
