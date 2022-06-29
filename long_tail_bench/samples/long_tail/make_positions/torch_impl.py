import torch
from long_tail_bench.core.executer import Executer


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) *
            mask).long() + padding_idx


def args_adaptor(np_args):
    tensor = torch.from_numpy(np_args[0]).cuda()
    padding_idx = 2
    return [tensor, padding_idx]


def executer_creator():
    return Executer(make_positions, args_adaptor)
