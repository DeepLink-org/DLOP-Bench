import torch
from bench.core.executer import Executer


def cdist(x1, x2, p_):
    output = torch.cdist(x1, x2, p=p_)
    return output


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x2 = torch.from_numpy(np_args[1]).cuda()
    p = np_args[2][0]
    return [x1, x2, p]


def executer_creator():
    return Executer(cdist, args_adaptor)
