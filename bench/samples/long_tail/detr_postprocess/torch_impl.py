# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from torch import nn
import torch.nn.functional as F
from bench.samples.long_tail.box_cxcywh_to_xyxy.torch_impl import (box_cxcywh_to_xyxy, )  # noqa
from bench.core.executer import Executer


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the
    coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the
                            size of each images of the batch
                          For evaluation, this must be the original image size
                            (before any data augmentation)
                          For visualization, this should be the image size
                            after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(torch.float32)
        boxes = boxes * scale_fct[:, None, :]

        results = [{
            "scores": s,
            "labels": l,
            "boxes": b
        } for s, l, b in zip(scores, labels, boxes)]
        return results


def args_adaptor(np_args):
    pred_logits = torch.from_numpy(np_args[0]).cuda()
    pred_boxes = torch.from_numpy(np_args[1]).cuda()
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    target_sizes = torch.from_numpy(np_args[2]).cuda()
    return [outputs, target_sizes]


def executer_creator():
    coder_instance = PostProcess()
    return Executer(coder_instance.forward, args_adaptor)
