import torch
from long_tail_bench.core.executer import Executer


def bbox_target_expand(bbox_targets, bbox_weights, labels, bbox_targets_expand,
                       bbox_weights_expand):
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand


def args_adaptor(np_args):
    bbox_targets = torch.from_numpy(np_args[0]).cuda()
    bbox_weights = torch.from_numpy(np_args[1]).cuda()
    labels = torch.from_numpy(np_args[2]).cuda()
    bbox_targets_expand = torch.from_numpy(np_args[3]).cuda()
    bbox_weights_expand = torch.from_numpy(np_args[4]).cuda()

    return [
        bbox_targets,
        bbox_weights,
        labels,
        bbox_targets_expand,
        bbox_weights_expand,
    ]


def executer_creator():
    return Executer(bbox_target_expand, args_adaptor)
