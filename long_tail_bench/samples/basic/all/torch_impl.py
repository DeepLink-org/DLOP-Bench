import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def all(input):
    input_np = np.random.random(input)
    input_tensor = torch.from_numpy(input_np).to(torch.float32).cuda()
    return torch.all(input_tensor)


def args_adaptor(np_args):
    input_tensor = np_args[0]
    return [input_tensor]


def executer_creator():
    return Executer(all, args_adaptor)
