import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class WBCELoss(nn.Module):
    label_keys = ["edge_target"]

    def __init__(self, loss_weight=10.0, reduction="mean"):
        super(WBCELoss, self).__init__()
        self.loss_weight = loss_weight
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def forward(self, edge_pred, edge_target, ignore_index=255, **kwargs):
        target = edge_target.unsqueeze(1)
        target[target == ignore_index] = 0
        target = target.float()
        beta = 1 - target.mean()
        label_weight = 1 - beta + (2 * beta - 1) * target
        loss = F.binary_cross_entropy_with_logits(edge_pred,
                                                  target,
                                                  weight=label_weight,
                                                  reduction=self.reduction)
        return self.loss_weight * loss

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}"
        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True
    label = torch.from_numpy(np_args[1]).cuda()
    label = torch.argmax(label, dim=1)
    return [logit, label]


def executer_creator():
    coder_instance = WBCELoss().cuda()
    return Executer(coder_instance.forward, args_adaptor)
