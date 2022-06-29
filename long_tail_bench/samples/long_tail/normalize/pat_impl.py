import torch
from long_tail_bench.core.executer import Executer


def normalize(src, mean, scale):
    # RGB to BGR
    dup = src.clone()
    dup[:, :, 0] = src[:, :, 2]
    dup[:, :, 2] = src[:, :, 0]
    return (dup - mean) * scale


def args_adaptor(np_args):
    img = torch.from_numpy(np_args[0]).cuda()
    mean = torch.from_numpy(np_args[1]).cuda()
    scale = torch.from_numpy(np_args[2]).cuda()

    return [img, mean, scale]


def executer_creator():
    return Executer(normalize, args_adaptor)
