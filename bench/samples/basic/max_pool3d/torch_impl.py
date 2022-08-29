import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def max_pool3d(input_image, kernel_size_image,\
        stride_image, padding_image, dilation_image, ceil_mode_image,\
        return_indices_image):
    ret = torch.nn.functional.max_pool3d(input_image, kernel_size_image,\
        stride_image, padding_image, dilation_image, ceil_mode_image,\
        return_indices_image)
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    kernel_size = np_args[1]
    stride = np_args[2]
    padding = np_args[3]
    dilation = np_args[4] 
    ceil_mode = np_args[5]
    return_indices =  np_args[6]
    if (len(kernel_size)) >1 :
        kernel_size_image = tuple(kernel_size)
    else:
        kernel_size_image = kernel_size[0]
    if (len(stride)) >1 :
        stride_image = tuple(stride)
    else:
        stride_image = stride[0]
    padding_image = padding[0]
    dilation_image = dilation[0]
    ceil_mode_image = ceil_mode[0]
    return_indices_image = return_indices[0]
    return [input_image, kernel_size_image,\
        stride_image, padding_image, dilation_image, ceil_mode_image,\
        return_indices_image]


def executer_creator():
    return Executer(max_pool3d, args_adaptor)
