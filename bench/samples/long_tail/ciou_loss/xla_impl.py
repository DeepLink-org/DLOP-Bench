# Copyright (c) OpenComputeLab. All Rights Reserved.
import numpy as np
import tensorflow as tf
import math
from bench.core.executer import Executer

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['giou', 'iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return tf.zeros([rows, 1]) if is_aligned else tf.zeros([rows, cols])

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    if is_aligned:
        lt = tf.math.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = tf.math.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = tf.clip_by_value((rb - lt), clip_value_min=0, clip_value_max=tf.float32.max)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]

        if mode == 'iou':
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = tf.math.minimum(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = tf.math.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    else:
        lt = tf.math.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = tf.math.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = tf.clip_by_value((rb - lt), clip_value_min=0, clip_value_max=tf.float32.max)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]

        if mode == 'iou' or mode == 'giou':
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]
        if mode == 'giou':
            enclosed_lt = tf.math.minimum(bboxes1[:, :, None, :2],
                                    bboxes2[:, None, :, :2])
            enclosed_rb = tf.math.maximum(bboxes1[:, :, None, 2:],
                                    bboxes2[:, None, :, 2:])

    eps = tf.constant([eps])
    union = tf.math.maximum(union, eps)
    ious = overlap / union

    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = tf.clip_by_value((enclosed_rb - enclosed_lt), clip_value_min=0, clip_value_max=tf.float32.max)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    enclose_area = tf.math.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def ciou_loss(pred, target, eps=1e-6):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = tf.math.maximum(pred[:, :2], target[:, :2])
    rb = tf.math.minimum(pred[:, 2:], target[:, 2:])
    wh = tf.clip_by_value((rb - lt), clip_value_min=0, clip_value_max=tf.float32.max)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = tf.math.minimum(pred[:, :2], target[:, :2])
    enclose_x2y2 = tf.math.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = tf.clip_by_value((enclose_x2y2 - enclose_x1y1), clip_value_min=0, clip_value_max=tf.float32.max)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    loss = 1 - cious
    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    # pred.requires_grad = True

    return [pred, target]


def executer_creator():
    return Executer(ciou_loss, args_adaptor)
