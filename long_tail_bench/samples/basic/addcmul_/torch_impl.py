import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def addcmul_(input_size, tensor1_size, tensor2_size):
    input_image_np = np.random.random([input_size, tensor1_size, tensor2_size])
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    ret = torch.addcmul_(input_image)
    return ret


def args_adaptor(np_args):
    input_size = np_args[0]
    tensor1_size = np_args[1]
    tensor2_size = np_args[2]
    
    return [input_size, tensor1_size, tensor2_size]


def executer_creator():
    return Executer(addcmul_, args_adaptor)
