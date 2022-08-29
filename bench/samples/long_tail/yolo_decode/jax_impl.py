# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax
import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer

def yolo_decode(bboxes, pred_bboxes, stride):
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    # Get outputs x, y
    x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
    y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
    w_pred = jnp.exp(pred_bboxes[..., 2]) * w
    h_pred = jnp.exp(pred_bboxes[..., 3]) * h

    decoded_bboxes = jnp.stack(
        (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
         x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
        axis=-1)

    return decoded_bboxes


def args_adaptor(np_args):
    bboxes = device_put(np_args[0])
    gt_bboxes = device_put(np_args[1])
    stride = device_put(np_args[2])

    return [bboxes, gt_bboxes, stride]


def executer_creator():
    return Executer(yolo_decode, args_adaptor)
