import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def meshgrid(*inputs):
    for input_image in inputs:
        input_image.requires_grad = True
    ret = torch.meshgrid(inputs)
    for i in range(len(inputs)):
        ret[i].backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_images = []
    for np_image in np_args:
        input_images.append(torch.from_numpy(np_image).to(torch.float32).cuda())

    return input_images

def executer_creator():
    return Executer(meshgrid, args_adaptor)
