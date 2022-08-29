# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def meshgrid(*inputs):
    ret = torch.meshgrid(inputs)
    for i in range(len(inputs)):
        ret[i].backward(torch.ones_like(ret[i]))
    return ret

def args_adaptor(np_args):
    input_images = []
    for np_image in np_args:
        input_images.append(torch.from_numpy(np_image).to(torch.float32).cuda())
    for input_image in input_images:
        input_image.requires_grad = True
    return input_images

def executer_creator():
    return Executer(meshgrid, args_adaptor)
