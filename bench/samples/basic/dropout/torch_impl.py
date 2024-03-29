# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def dropout(input_image, p, inplace):
    dropout = torch.nn.Dropout(p=p, inplace=inplace)
    ret = dropout(input_image)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    p = np_args[1]
    inplace = np_args[2]
    
    return [input_image, p, inplace]

def executer_creator():
    return Executer(dropout, args_adaptor)
