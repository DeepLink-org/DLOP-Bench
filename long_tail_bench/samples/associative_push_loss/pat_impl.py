import torch
from long_tail_bench.core.executer import Executer


def forward(input, mask, reduction="mean", normalizer=None):
    emb0, emb1 = input
    mask_f = mask.float()
    num = mask_f.sum(dim=1, keepdim=True)
    emb0 = emb0.float().squeeze(-1)
    emb1 = emb1.float().squeeze(-1)
    emb_mean = (emb0 + emb1) / 2
    mask_cross = mask_f.unsqueeze(1) + mask_f.unsqueeze(2)
    mask = mask_cross.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = emb_mean.unsqueeze(1) - emb_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = torch.nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return push


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
