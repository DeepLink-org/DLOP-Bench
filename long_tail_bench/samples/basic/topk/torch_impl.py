import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def topk(topk_0, topk_1):
    return torch.topk(topk_0, topk_1)

def args_adaptor(np_args):
    topk_0 = torch.from_numpy(np_args[0]).cuda()
    topk_1 = torch.from_numpy(np_args[1]).cuda()
    return [topk_0, topk_1]


def executer_creator():
    return Executer(topk, args_adaptor)
