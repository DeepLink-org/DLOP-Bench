import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def argmax(input_size, dim):
    input_image_np = np.random.random(input_size)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    return torch.argmax(input_image, dim=dim)


def args_adaptor(np_args):
    input_size = np_args[0]
    dim = np_args[1][0]
    return [input_size, dim]


def executer_creator():
    return Executer(argmax, args_adaptor)
