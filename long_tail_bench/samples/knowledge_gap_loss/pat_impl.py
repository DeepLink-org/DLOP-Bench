import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class KNOWLEDGEGAPLoss(nn.Module):
    """L2Loss.

    Args:
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    label_keys = ["gt_semantic_seg", "gt_soft_seg"]

    def __init__(self, loss_weight=1.0, reduction="mean", temp=1):
        super().__init__()
        self.loss_weight = loss_weight
        self.T = temp
        self.reduction = reduction

    def forward(self, cls_score, label, soft_label, **kwargs):
        """Forward function."""
        soft_label = soft_label.type_as(cls_score).detach()
        if soft_label.shape != cls_score.shape:
            soft_label = F.interpolate(
                soft_label,
                size=cls_score.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        assert soft_label.dim() == cls_score.dim()
        index = label != 255
        num_classes = soft_label.shape[1]
        label[label == 255] = 0
        # figure p_s
        cls_score = cls_score / self.T
        tmp_cls_score = F.softmax(cls_score, dim=1).detach()
        # figure p_t
        tmp_soft_label = F.softmax(soft_label / self.T, dim=1)
        # figure wn
        soft_label_correct = torch.gather(tmp_soft_label, 1,
                                          label.unsqueeze(1)).squeeze(1)
        cls_score_correct = torch.gather(tmp_cls_score, 1,
                                         label.unsqueeze(1)).squeeze(1)
        w = soft_label_correct - cls_score_correct
        w = torch.clamp(w, min=0)
        w = w.unsqueeze(1)
        w = w.expand(-1, num_classes, -1, -1)
        # figure h_soft
        h_soft = tmp_soft_label * (tmp_soft_label.log() -
                                   F.log_softmax(cls_score, dim=1))
        h_soft = w * h_soft
        h_soft_total = h_soft.sum(dim=1)
        h_soft_total = h_soft_total[index].mean()

        loss_cls = h_soft_total * self.loss_weight
        return loss_cls

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, temperature={self.T}"
        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True

    label = torch.from_numpy(np_args[1]).cuda()
    label = torch.argmax(label, dim=1)

    softlabel = torch.from_numpy(np_args[2]).cuda()

    return [logit, label, softlabel]


def executer_creator():
    coder_instance = KNOWLEDGEGAPLoss().cuda()
    return Executer(coder_instance.forward, args_adaptor)
