import torch
import torch.nn
from bench.core.executer import Executer

def log_softmax(log_softmax_0, log_softmax_1):
    return torch.log_softmax(log_softmax_0, log_softmax_1)

def args_adaptor(np_args):
    log_softmax_0 = torch.from_numpy(np_args[0]).cuda()
    log_softmax_1 = torch.from_numpy(np_args[1]).cuda()
    return [log_softmax_0, log_softmax_1]


def executer_creator():
    return Executer(log_softmax, args_adaptor)
