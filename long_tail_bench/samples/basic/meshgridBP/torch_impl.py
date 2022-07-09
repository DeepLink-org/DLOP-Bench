import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def meshgrid(*inputs):
    input_images = []
    for input_size in inputs:
        input_image_np = np.random.random(input_size)
        input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
        input_image.requires_grad = True
        input_images.append(input_image)
    ret = torch.meshgrid(input_images)
    for i in range(len(inputs)):
        ret[i].backward(ret)
    return ret

def args_adaptor(np_args):
    return np_args


def executer_creator():
    return Executer(meshgrid, args_adaptor)
