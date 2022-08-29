# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def eye(n, m):
    ret = torch.eye(n, m, device="cuda")
    ret.requires_grad=True
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    return [np_args[0], np_args[1]]


def executer_creator():
    return Executer(eye, args_adaptor)
