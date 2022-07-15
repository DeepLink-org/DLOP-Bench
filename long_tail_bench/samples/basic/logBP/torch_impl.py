import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def log(log_0):
    log.requires_grad = True
    ret = torch.log(log_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    log_0 = torch.from_numpy(np_args[0])
    return [log_0]


def executer_creator():
    return Executer(log, args_adaptor)
