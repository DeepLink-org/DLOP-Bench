import torch
import torch.nn
from bench.core.executer import Executer

def lt(lt_0, lt_1):
    return torch.lt(lt_0, lt_1)

def args_adaptor(np_args):
    lt_0 = torch.from_numpy(np_args[0]).cuda()
    lt_1 = torch.from_numpy(np_args[1]).cuda()
    return [lt_0, lt_1]


def executer_creator():
    return Executer(lt, args_adaptor)
