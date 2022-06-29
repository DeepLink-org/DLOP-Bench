import torch
from long_tail_bench.core.executer import Executer


def sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size.
    Also converts from relative to absolute coordinates
    and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 *= img_size
    _x2 *= img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)
    return x1, x2


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x2 = torch.from_numpy(np_args[1]).cuda()
    return [x1, x2, 256, 0, True]


def executer_creator():
    return Executer(sanitize_coordinates, args_adaptor)
