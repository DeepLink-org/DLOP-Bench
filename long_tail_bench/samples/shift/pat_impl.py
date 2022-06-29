import torch
from long_tail_bench.core.executer import Executer


def shift(x, num_segments, shift_div=3):
    """Perform temporal shift operation on the feature.

    Args:
        x (torch.Tensor): The input feature to be shifted.
        num_segments (int): Number of frame segments.
        shift_div (int): Number of divisions for shift. Default: 3.

    Returns:
        torch.Tensor: The shifted feature.
    """
    # [N, C, H, W]
    n, c, h, w = x.size()

    # [N // num_segments, num_segments, C, H*W]
    # can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(-1, num_segments, c, h * w)

    # get shift fold
    fold = c // shift_div

    # split c channel into three parts:
    # left_split, mid_split, right_split
    left_split = x[:, :, :fold, :]
    mid_split = x[:, :, fold:2 * fold, :]
    right_split = x[:, :, 2 * fold:, :]

    # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
    # because array on caffe inference must be got by computing

    # shift left on num_segments channel in `left_split`
    zeros = left_split - left_split
    blank = zeros[:, :1, :, :]
    left_split = left_split[:, 1:, :, :]
    left_split = torch.cat((left_split, blank), 1)

    # shift right on num_segments channel in `mid_split`
    zeros = mid_split - mid_split
    blank = zeros[:, :1, :, :]
    mid_split = mid_split[:, :-1, :, :]
    mid_split = torch.cat((blank, mid_split), 1)

    # right_split: no shift

    # concatenate
    out = torch.cat((left_split, mid_split, right_split), 2)

    # [N, C, H, W]
    # restore the original dimension
    return out.view(n, c, h, w)


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    num_segments = np_args[1]
    shift_div = np_args[2]
    x.reuqires_grad = True

    return [x, num_segments, shift_div]


def executer_creator():
    return Executer(shift, args_adaptor)
