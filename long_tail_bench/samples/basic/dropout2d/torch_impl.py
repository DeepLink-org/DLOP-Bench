import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def dropout2d(input_size, p, training, inplace):
    input_image_np = np.random.random([input_size])
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    dropout2d = torch.nn.dropout2d(p=p, training=training, inplace=inplace)
    ret = dropout2d(input_image)
    return ret


def args_adaptor(np_args):
    input_size = np_args[0]
    p = np_args[1][0]
    training = np_args[2][0]
    inplace = np_args[3][0]
    
    return [input_size, p, training, inplace]

def executer_creator():
    return Executer(dropout2d, args_adaptor)
