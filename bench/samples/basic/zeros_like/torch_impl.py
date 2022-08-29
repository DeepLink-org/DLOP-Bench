# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np
from bench.core.executer import Executer


dtype_mapping = {
    "bool": torch.bool,
    "float32": torch.float32, 
    "float": torch.float, 
    "float64": torch.float64, 
    "double": torch.double, 
    "float16": torch.float16, 
    "bfloat16": torch.bfloat16, 
    "half": torch.half, 
    "uint8": torch.uint8, 
    "int8": torch.int8, 
    "int16": torch.int16, 
    "short": torch.short, 
    "int32": torch.int32, 
    "int": torch.int, 
    "int64": torch.int64, 
    "long": torch.long
}

def zeros_like(input_image):
    ret = torch.zeros_like(input_image, device=torch.device("cuda"), dtype=input_image.dtype)
    return ret

def args_adaptor(np_args):
    size = np_args[0]
    dtype_str = np_args[1]
    dtype = None
    if dtype_str != "None":
        dtype = dtype_mapping[dtype_str]
    np_input = np.random.random(size)
    input_image = torch.from_numpy(np_input).to(dtype).cuda()
    return [input_image]

def executer_creator():
    return Executer(zeros_like, args_adaptor)