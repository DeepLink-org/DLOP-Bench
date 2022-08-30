# Copyright (c) OpenComputeLab. All Rights Reserved.

import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(a_min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def args_adaptor(np_args):
    boxes1 = device_put(np_args[0])
    boxes2 = device_put(np_args[1])
    return [boxes1, boxes2]


def executer_creator():
    return Executer(box_iou, args_adaptor)
