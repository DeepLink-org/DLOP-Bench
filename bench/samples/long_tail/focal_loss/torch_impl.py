# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch
import torch.nn as nn
import torch.nn.functional as F
from bench.core.executer import Executer


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, ignore_index=255, **kwargs):
        ce_loss = F.cross_entropy(
            cls_score,
            label,
            weight=None,
            reduction="none",
            ignore_index=ignore_index,
        )
        pt = torch.exp(-ce_loss)
        loss = (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()
        return self.loss_weight * loss

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, alpha={self.alpha}, \
              gamma={self.gamma}"

        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True
    label = torch.from_numpy(np_args[1]).cuda()
    label = torch.argmax(label, dim=1)
    return [logit, label]


def executer_creator():
    coder_instance = FocalLoss().cuda()
    return Executer(coder_instance.forward, args_adaptor)
