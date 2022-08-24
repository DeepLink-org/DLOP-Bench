import torch
import torch.nn
from bench.core.executer import Executer

def erfinv_(erfinv__0):
    return erfinv__0.erfinv_()

def args_adaptor(np_args):
    erfinv__0 = torch.from_numpy(np_args[0]).cuda()
    return [erfinv__0]


def executer_creator():
    return Executer(erfinv_, args_adaptor)
