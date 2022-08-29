# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch
from torch import nn
import torch.nn.functional as F
from bench.core.executer import Executer


@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
            outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
            (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special
                no-object category
            matcher: module able to compute a matching between targets and
                proposals
            weight_dict: dict containing as key the names of the losses and
                as values their relative weight.
            eos_coef: relative classification weight applied to the no-object
                category
            losses: list of all the losses to be applied. See get_loss for
                list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1).cuda()
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim
            [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this
            # one here
            losses["class_error"] = (
                100 - accuracy(src_logits[idx], target_classes_o)[0])
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def args_generator(num_target_boxes):
    num_classes = 3
    batch_size, num_queries = 4, 100
    pred_logits = torch.randn([batch_size, num_queries, num_classes],
                              device="cuda")
    pred_boxes = torch.randn([batch_size, num_queries, 4], device="cuda")
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = []
    for _ in range(batch_size):
        labels = torch.randn([num_target_boxes], device="cuda")
        boxes = torch.randn([num_target_boxes, 4], device="cuda")
        target = {"labels": labels, "boxes": boxes}
        targets.append(target)

    indices_ = [
        ([1, 6, 18, 34, 49, 50, 52, 98], [6, 0, 4, 7, 1, 2, 3, 5]),
        ([3, 5], [0, 1]),
        ([33, 54], [1, 0]),
        ([90], [0]),
    ]
    indices = [(
        torch.as_tensor(i, dtype=torch.int64),
        torch.as_tensor(j, dtype=torch.int64),
    ) for i, j in indices_]
    return [outputs, targets, indices, num_target_boxes]


def executer_creator():
    coder_instance = SetCriterion(num_classes=2,
                                  matcher=None,
                                  weight_dict=None,
                                  eos_coef=0.1,
                                  losses=None)
    return Executer(coder_instance.loss_labels, args_generator)
