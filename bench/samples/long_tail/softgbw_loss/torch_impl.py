# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bench.core.executer import Executer


def get_softgauss_loss_weight(logit, target, sigma=7, dis=1.5):
    pred = torch.argmax(logit, dim=1)
    temp_target = target.data.cpu().clone().numpy()
    temp_pred = pred.data.cpu().clone().numpy()
    num_images = len(temp_target)
    heatmaps = np.zeros_like(temp_target).astype(np.float)
    for i in range(num_images):
        tmp_target = temp_target[i]
        tmp_pred = temp_pred[i]
        ignore_idx = tmp_target == 255
        tmp_target[ignore_idx] = 0
        tmp_pred[ignore_idx] = 0
        hard_idx = tmp_target != tmp_pred
        edge = cv2.Canny(np.uint8(tmp_target), 1, 1)
        edge[hard_idx] = 255
        dist = cv2.distanceTransform(255 - edge, cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)
        heatmap = np.ones_like(dist)
        heatmap = heatmap + np.exp(-1 * np.square(dist) / (2 * sigma**2))
        heatmap[ignore_idx] = 0
        heatmap = heatmap / (np.sum(heatmap) + 1e-10)
        heatmaps[i] = heatmap
    return torch.from_numpy(heatmaps)


class SoftGBWLoss(nn.Module):
    def __init__(self, sigma=7, dis=1.5, loss_weight=1.0):
        super(SoftGBWLoss, self).__init__()
        self.sigma = sigma
        self.dis = dis
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, ignore_index=255, **kwargs):
        gbdweight = get_softgauss_loss_weight(cls_score,
                                              label,
                                              sigma=self.sigma,
                                              dis=self.dis)
        gbdweight = gbdweight.type_as(cls_score)
        ce_loss = F.cross_entropy(
            cls_score,
            label,
            weight=None,
            reduction="none",
            ignore_index=ignore_index,
        )
        loss = ce_loss * gbdweight
        loss = loss.sum((1, 2))
        loss = loss.mean()

        return self.loss_weight * loss

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, \
              sigma={self.sigma}, dis={self.dis}"

        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    label = torch.argmax(torch.from_numpy(np_args[1]).cuda(), dim=1)
    logit.requires_grad = True

    return [logit, label]


def executer_creator():
    coder_instance = SoftGBWLoss()
    return Executer(coder_instance.forward, args_adaptor)
