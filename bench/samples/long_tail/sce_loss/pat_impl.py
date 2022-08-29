import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class SCELoss(nn.Module):
    """L2Loss.

    Args:
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    label_keys = ["gt_semantic_seg", "gt_soft_seg"]

    def __init__(self,
                 loss_weight=1.0,
                 reduction="mean",
                 temp=1,
                 channel_sum="False"):
        super().__init__()
        self.loss_weight = loss_weight
        self.T = temp
        self.reduction = reduction
        self.channel_sum = True if channel_sum == "True" else False

    def forward(self, cls_score, gt_semantic_seg, soft_label, **kwargs):
        """Forward function."""
        # make sure the score and label is same shape
        soft_label = soft_label.type_as(cls_score).detach()
        if soft_label.shape != cls_score.shape:
            soft_label = F.interpolate(
                soft_label,
                size=cls_score.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        assert soft_label.dim() == cls_score.dim()
        score_size = cls_score.size()
        nclass = cls_score.shape[1]
        # record the ignore region
        ignore_index = (gt_semantic_seg == 255).unsqueeze(1)
        # record the valid region
        valid_index = gt_semantic_seg != 255
        # record the true pred region and false pred region
        soft2hard = torch.argmax(soft_label, dim=1)
        # ture_index = (gt_semantic_seg == soft2hard).\
        # unsqueeze(1).expand(score_size)
        false_index = ((gt_semantic_seg !=
                        soft2hard).unsqueeze(1).expand(score_size))
        # generate one hot label
        gt_onehot = (torch.zeros(score_size).type_as(gt_semantic_seg) +
                     0.05 / nclass)
        gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        gt_semantic_seg[ignore_index] = 0
        gt_onehot = gt_onehot.scatter(1, gt_semantic_seg, 0.95)

        # figure fusion weight
        t_prob = F.softmax(soft_label / self.T, dim=1)
        t_prob_sort = torch.sort(t_prob, dim=1, descending=True)[0]
        t_conf = t_prob_sort[:, 0:1] - t_prob_sort[:, 1:2]

        # generate soft label
        tmp_label = t_prob * t_conf + gt_onehot * (1 - t_conf)
        t_prob[false_index] = tmp_label[false_index]
        t_prob = t_prob.detach()

        tmp_cls_score = F.log_softmax(cls_score / self.T, dim=1)

        # h_soft = -1 * (tmp_label * tmp_cls_score)
        h_soft = t_prob * (t_prob.log() - tmp_cls_score)
        if self.channel_sum:
            h_soft_total = h_soft.sum(dim=1)
        else:
            h_soft_total = h_soft.mean(dim=1)
        if not torch.any(valid_index):
            return h_soft_total.mean() * 0
        h_soft_total = h_soft_total[valid_index].mean()
        loss_cls = self.loss_weight * h_soft_total

        # tmp = F.kl_div(F.log_softmax(cls_score / self.T, dim=1),
        #                F.softmax(label / self.T, dim=1), reduction='mean')
        return loss_cls

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, temperature={self.T}, \
              channel_sum={self.channel_sum}"

        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    label = torch.argmax(torch.from_numpy(np_args[1]).cuda(), dim=1)
    softlabel = torch.from_numpy(np_args[2]).cuda()
    logit.requires_grad = True

    return [logit, label, softlabel]


def executer_creator():
    coder_instance = SCELoss(channel_sum="True")
    return Executer(coder_instance.forward, args_adaptor)
