import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def meshgrid(*inputs):
    input_images = []
    for input_size in inputs:
        input_image_np = np.random.random(input_size)
        input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
        input_images.append(input_image)
    return torch.meshgrid(input_images)


def args_adaptor(np_args):
    print(np_args)
    return np_args


def executer_creator():
    return Executer(meshgrid, args_adaptor)
