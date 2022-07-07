import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def pad(input, pad):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    pad_image = tuple(pad)
    ret = torch.nn.functional.pad(input_image, pad_image).cuda()
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    pad = np_args[1]
    return [input, pad]


def executer_creator():
    return Executer(pad, args_adaptor)
