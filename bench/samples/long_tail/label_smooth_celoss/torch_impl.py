# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from bench.core.executer import Executer


class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio=0.1, num_classes=6):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)

        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.v)

        loss = -torch.sum(F.log_softmax(input, 1) *
                          (one_hot.detach())) / input.size(0)  # noqa
        return loss


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    boxes.requires_grad = True

    label = torch.from_numpy(np_args[1]).cuda()

    return [boxes, label]


def executer_creator():
    coder_instance = LabelSmoothCELoss()
    return Executer(coder_instance.forward, args_adaptor)
