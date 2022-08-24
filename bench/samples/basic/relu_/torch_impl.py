import torch
import torch.nn
from bench.core.executer import Executer

def relu_(relu__0):
    return relu__0.relu_()

def args_adaptor(np_args):
    relu__0 = torch.from_numpy(np_args[0]).cuda()
    return [relu__0]


def executer_creator():
    return Executer(relu_, args_adaptor)
