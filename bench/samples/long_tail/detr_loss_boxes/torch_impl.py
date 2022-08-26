# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from torch import nn
import torch.nn.functional as F
from bench.core.executer import Executer
from bench.samples.long_tail.generalized_box_iou.torch_impl import (
    generalized_box_iou, )  # noqa
from bench.samples.long_tail.box_cxcywh_to_xyxy.torch_impl import (
    box_cxcywh_to_xyxy, )  # noqa


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
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "boxes"
        containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h),
        normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
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
    coder_instance = SetCriterion(num_classes=3,
                                  matcher=None,
                                  weight_dict=None,
                                  eos_coef=0.1,
                                  losses=None)  # noqa
    return Executer(coder_instance.loss_boxes, args_generator)
