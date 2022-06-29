import torch
from long_tail_bench.core.executer import Executer


def squash(Z):
    """squash
    """
    vec_squared_norm = torch.sum(torch.square(Z), axis=-1, keepdim=True)
    scalar_factor = vec_squared_norm / \
        (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * Z
    return vec_squashed


def args_adaptor(np_args):
    input0 = torch.from_numpy(np_args[0]).cuda()
    return [input0]


def executer_creator():
    return Executer(squash, args_adaptor)
