import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def eye(n, m):
    return torch.eye(n, m, device="cuda")


def args_adaptor(np_args):
    return [np_args[0], np_args[1]]


def executer_creator():
    return Executer(eye, args_adaptor)
