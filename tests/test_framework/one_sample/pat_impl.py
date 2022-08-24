import torch
from bench.core.executer import Executer
from tests.test_framework.one_sample.common import count_plus_one


def func(a, b):
    count_plus_one()
    return a + b


def args_adaptor(np_args):
    a = torch.from_numpy(np_args[0]).cuda()
    b = torch.from_numpy(np_args[1]).cuda()
    return [a, b]


def executer_creator():
    return Executer(func, args_adaptor)
