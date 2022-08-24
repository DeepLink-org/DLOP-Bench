import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return tf.stack([cx, cy, w, h], dim=1)
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
    dw = tf.clip_by_value(dw, dw, np.log(1000. / 16.))
    dh = tf.clip_by_value(dh, dh, np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = tf.exp(dw) * widths[:, None]
    pred_h = tf.exp(dh) * heights[:, None]

    pred_boxes = tf.Variable(tf.zeros(offset.shape))
    # x1
    pred_boxes[:, 0::4].assign(pred_ctr_x - 0.5 * pred_w)
    # y1
    pred_boxes[:, 1::4].assign(pred_ctr_y - 0.5 * pred_h)
    # x2
    pred_boxes[:, 2::4].assign(pred_ctr_x + 0.5 * pred_w - 1)
    # y2
    pred_boxes[:, 3::4].assign(pred_ctr_y + 0.5 * pred_h - 1)

    pred_boxes = tf.convert_to_tensor(pred_boxes)

    return pred_boxes


def args_adaptor(np_args):
    boxes = tf.convert_to_tensor(np_args[0], tf.float32)
    offset = tf.convert_to_tensor(np_args[1], tf.float32)
    weights = [1.0, 1.0, 1.0, 1.0]

    return [boxes, offset, weights]


def executer_creator():
    return Executer(offset2bbox, args_adaptor)
