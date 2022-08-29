import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def rsub(rsub_0, rsub_1):
    return torch.rsub(rsub_0, rsub_1)

def args_adaptor(np_args):
    rsub_0 = torch.from_numpy(np_args[0]).cuda()
    rsub_1 = torch.from_numpy(np_args[1]).cuda()
    return [rsub_0, rsub_1]


def executer_creator():
    return Executer(rsub, args_adaptor)
