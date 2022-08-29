# Copyright (c) OpenComputeLab. All Rights Reserved.

from math import ceil
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def max_unpool2d(input_size_image, indices_image, kernel_size_image, \
        stride_image, padding_image, output_size_image):

    ret = torch.nn.functional.max_unpool2d(input_size_image, indices_image, kernel_size_image, \
        stride_image, padding_image).cuda()
    ret.backward(torch.ones_like(ret))

    return ret

def args_adaptor(np_args):
    input_i = np_args[0]
    indices_i = np_args[1]
    kernel_size = np_args[2]
    stride = np_args[3]
    padding = np_args[4]
    output_image = np_args[5]

    if (len(kernel_size)) >1 :
        kernel_size_image = tuple(kernel_size)
    else:
        kernel_size_image = kernel_size[0]
    if (len(stride)) >1 :
        stride_image = tuple(stride)
    else:
        stride_image = stride[0]
    if (len(padding)) >1:
        padding_image = padding[0]
    else:
        padding_image = padding[0]

    pool_input_image_np = np.random.random(input_i)
    pool_input_image = torch.from_numpy(pool_input_image_np).to(torch.float32).cuda()
    pool_input_image.requires_grad = True
    pool = torch.nn.MaxPool2d(kernel_size_image,stride_image,padding_image, return_indices=True)
    input_size_image, indices_image = pool(pool_input_image)
    input_size_image.requires_grad = True
    input_size_image.backward(input_size_image)
    indices_image.requires_grad = True
    indices_image.backward(indices_image)

    return [input_size_image, indices_i, kernel_size_image, \
        stride_image, padding_image, output_image]

def executer_creator():
    return Executer(max_unpool2d, args_adaptor)
