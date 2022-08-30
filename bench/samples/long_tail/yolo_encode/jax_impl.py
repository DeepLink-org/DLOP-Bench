# Copyright (c) OpenComputeLab. All Rights Reserved.

import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer

def yolo_encode(bboxes, gt_bboxes, stride, eps=1e-6):
    x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
    y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
    w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    w_target = jnp.log(w_gt / w)
    w_target = jnp.clip(eps, w_target)
    h_target = jnp.log(h_gt / h)
    h_target = jnp.clip(eps, h_target)
    x_center_target = ((x_center_gt - x_center) / stride + 0.5)
    y_center_target = ((y_center_gt - y_center) / stride + 0.5)
    x_center_target = jnp.clip(x_center_target, eps, 1 - eps)
    y_center_target = jnp.clip(y_center_target, eps, 1 - eps)
    encoded_bboxes = jnp.stack(
        [x_center_target, y_center_target, w_target, h_target], axis=-1)
    return encoded_bboxes


def args_adaptor(np_args):
    bboxes = device_put(np_args[0])
    gt_bboxes = device_put(np_args[1])
    stride = device_put(np_args[2])
    eps = 1e-6

    return [bboxes, gt_bboxes, stride, eps]


def executer_creator():
    return Executer(yolo_encode, args_adaptor)
