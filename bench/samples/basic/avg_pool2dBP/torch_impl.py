import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def avg_pool2d(input_image, kernel_size_image,\
        stride_image, padding_image, ceil_mode_image, count_include_pad_image,\
        divisor_override_image):
    ret = torch.nn.functional.avg_pool2d(input_image, kernel_size_image,\
        stride_image, padding_image, ceil_mode_image, count_include_pad_image,\
        divisor_override_image)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_image_np = np_args[0]
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image.requires_grad = True
    kernel_size = np_args[1]
    stride = np_args[2]
    padding = np_args[3]
    ceil_mode = np_args[4]
    count_include_pad = np_args[5]
    divisor_override =  np_args[6]
    if (len(kernel_size)) >1:
        kernel_size_image = tuple(kernel_size)
    else:
        kernel_size_image = kernel_size[0]
    if (len(stride)) >1:
        stride_image = tuple(stride)
    else:
        stride_image = stride[0]
    if (len(padding)) >1:
        padding_image = tuple(padding)
    else:
        padding_image = padding[0]
    ceil_mode_image = ceil_mode[0]
    count_include_pad_image = count_include_pad[0]
    divisor_override_image = divisor_override[0]
    return [input_image, kernel_size_image,\
        stride_image, padding_image, ceil_mode_image, count_include_pad_image,\
        divisor_override_image]

def executer_creator():
    return Executer(avg_pool2d, args_adaptor)
