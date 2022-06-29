import torch
from long_tail_bench.core.executer import Executer


def forward(input, mask, reduction="mean", normalizer=None):
    emb0, emb1 = input
    mask_f = mask.float()
    num = mask_f.sum(dim=1, keepdim=True)
    emb0 = emb0.float().squeeze(-1)
    emb1 = emb1.float().squeeze(-1)
    emb_mean = (emb0 + emb1) / 2
    emb0 = torch.pow(emb0 - emb_mean, 2) / (num + 1e-4)
    emb0 = emb0[mask].sum()
    emb1 = torch.pow(emb1 - emb_mean, 2) / (num + 1e-4)
    emb1 = emb1[mask].sum()
    pull = emb0 + emb1
    return pull


def args_adaptor(np_args):
    boxes1 = torch.from_numpy(np_args[0]).cuda()
    boxes1.requires_grad = True
    boxes2 = torch.from_numpy(np_args[1]).cuda()
    boxes2.requires_grad = True
    boxes = [boxes1, boxes2]
    mask = torch.from_numpy(np_args[2]).cuda()
    return [boxes, mask]


def executer_creator():
    return Executer(forward, args_adaptor)
