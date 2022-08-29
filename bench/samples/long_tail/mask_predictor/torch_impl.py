# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import cv2
import torch
import numpy as np
from torch.nn.modules.utils import _pair
from bench.core.executer import Executer


class MaskPredictor(object):
    def __init__(self, num_classes=6, mask_thresh=0.5, share_location=False):
        self.num_classes = num_classes
        self.mask_thresh = mask_thresh
        self.share_location = share_location

    def predict(self, rois, heatmap, input):
        """
        Arguments:
            - rois (FloatTensor, fp32): [R, >=7] (batch_index, x1, y1, x2, y2,
            socre, cls)
            - heatmap (FloatTensor, fp32): [R, num_cls, label_h, label_w]
            - image_info (list of FloatTensor): [B, >=3] (image_h, image_w,
            scale_factor, ...)
        """
        image_info = input["image_info"]
        R = len(rois)
        B = len(image_info)

        rois = rois.detach().cpu().numpy()
        heatmap = heatmap.detach().cpu().numpy()
        # TODO: support share location without repeat
        if self.share_location:
            heatmap = heatmap.repeat(self.num_classes, axis=1)

        masks = [None] * R
        for b_ix in range(B):
            scale_h, scale_w = _pair(image_info[b_ix][2])
            img_h, img_w = map(int, image_info[b_ix][3:5])
            keep_inds = np.where(rois[:, 0] == b_ix)[0]
            img_rois = rois[keep_inds]
            img_heatmap = heatmap[keep_inds]
            img_rois[:, 1] = img_rois[:, 1] / scale_w
            img_rois[:, 2] = img_rois[:, 2] / scale_h
            img_rois[:, 3] = img_rois[:, 3] / scale_w
            img_rois[:, 4] = img_rois[:, 4] / scale_h

            for idx, (roi, htmap) in enumerate(zip(img_rois, img_heatmap)):
                x1, y1, x2, y2, score, cls = map(int, roi[1:7])
                roi_w = np.maximum(x2 - x1 + 1, 1)
                roi_h = np.maximum(y2 - y1 + 1, 1)
                mask = cv2.resize(htmap[cls], (roi_w, roi_h))
                mask = np.array(mask > self.mask_thresh, dtype=np.uint8)
                # mask = cv2.resize(cv2.UMat(htmap[cls]), (roi_w, roi_h))
                # mask = np.array(mask.get().astype('f') > self.mask_thresh,
                # dtype=np.uint8)
                img = np.zeros((img_h, img_w), dtype=np.uint8)
                img[y1:y1 + roi_h, x1:x1 + roi_w] = mask
                masks[keep_inds[idx]] = img
        return {"dt_masks": masks}


def args_adaptor(np_args):
    rois = torch.from_numpy(np_args[0]).cuda()
    heatmap = torch.from_numpy(np_args[1]).cuda()

    return [rois, heatmap, np_args[2]]


def executer_creator():
    coder_instance = MaskPredictor()
    return Executer(coder_instance.predict, args_adaptor)
