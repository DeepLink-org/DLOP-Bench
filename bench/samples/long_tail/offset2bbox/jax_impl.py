# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax
import jax.numpy as jnp
from jax import jit, device_put
from bench.core.executer import Executer

def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return jnp.stack((cx, cy, w, h), axis=1)
    else:
        return cx, cy, w, h

def offset2bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = jax.lax.max(dw, jnp.log(1000. / 16.))
    dh = jax.lax.max(dh, jnp.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = jnp.exp(dw) * widths[:, None]
    pred_h = jnp.exp(dh) * heights[:, None]

    pred_boxes = jnp.zeros(offset.shape)
    pred_boxes = pred_boxes.at[:, 0::4].set(pred_ctr_x - 0.5 * pred_w)
    pred_boxes = pred_boxes.at[:, 1::4].set(pred_ctr_x - 0.5 * pred_h)
    pred_boxes = pred_boxes.at[:, 2::4].set(pred_ctr_x - 0.5 * pred_w - 1)
    pred_boxes = pred_boxes.at[:, 3::4].set(pred_ctr_x - 0.5 * pred_h - 1)

    return pred_boxes


def args_adaptor(np_args):
    boxes = device_put(np_args[0])
    offset = device_put(np_args[1])
    weights = device_put(np_args[2])

    return [boxes, offset, weights]


def executer_creator():
    return Executer(offset2bbox, args_adaptor)
