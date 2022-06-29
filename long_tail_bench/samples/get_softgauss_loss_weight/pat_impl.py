import torch
import numpy as np
import cv2
from long_tail_bench.core.executer import Executer


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


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    return [logit, target]


def executer_creator():
    return Executer(get_softgauss_loss_weight, args_adaptor)
