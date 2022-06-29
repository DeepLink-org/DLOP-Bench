import torch
from long_tail_bench.core.executer import Executer


def index2d(src, idx):
    """
    Indexes a tensor by a 2d index.

    In effect, this does
        out[i, j] = src[i, idx[i, j]]
    Both src and idx should have the same size.
    """

    offs = torch.arange(idx.size(0), device=idx.device)[:, None].expand_as(idx)
    idx = idx + offs * idx.size(1)

    return src.view(-1)[idx.view(-1)].view(idx.size())


def args_adaptor(np_args):
    src = torch.from_numpy(np_args[0]).cuda()
    idx = torch.from_numpy(np_args[1]).cuda()

    return [src, idx]


def executer_creator():
    return Executer(index2d, args_adaptor)
