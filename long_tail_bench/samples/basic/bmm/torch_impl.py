import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def bmm(input, mat2):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    mat2_image_np = np.random.random(mat2)
    mat2_image = torch.from_numpy(mat2_image_np).to(torch.float32).cuda()
    ret = torch.bmm(input_image, mat2_image).cuda()
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    mat2 = np_args[1]
    return [input, mat2]


def executer_creator():
    return Executer(bmm, args_adaptor)
