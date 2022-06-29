import torch
from long_tail_bench.core.executer import Executer


def bbox2delta(proposals,
               gt,
               means=(0.0, 0.0, 0.0, 0.0),
               stds=(1.0, 1.0, 1.0, 1.0)):
    assert proposals.size() == gt.size()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()

    means = tuple(np_args[2].tolist())
    stds = tuple(np_args[3].tolist())
    return [proposals, gt, means, stds]


def executer_creator():
    return Executer(bbox2delta, args_adaptor)
