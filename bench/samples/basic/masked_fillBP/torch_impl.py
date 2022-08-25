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


def masked_fillBP(input_tensor, masked_tensor, value):
    input_tensor.requires_grad = True
    ret = input_tensor.masked_fill(masked_tensor, value)
    ret = ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_size = np_args[0]
    masked_size = np_args[1]
    value = np_args[2]
    dtype = None
    if isinstance(value, str) and "inf" in value:
        value = float(value)
        dtype = torch.float
    else:
        dtype = dtype_mapping.get(str(type(value)), None)
    input_np = np.random.random(input_size)
    input_tensor = torch.from_numpy(input_np).to(dtype).cuda()
    masked_np = np.random.randint(0, 2, size=masked_size) == 1
    masked_tensor = torch.from_numpy(masked_np).cuda()
    return [input_tensor, masked_tensor, value]

def executer_creator():
    return Executer(masked_fillBP, args_adaptor)
