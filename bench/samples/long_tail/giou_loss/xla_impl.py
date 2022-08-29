# Copyright (c) OpenComputeLab. All Rights Reserved.
import numpy as np
import tensorflow as tf
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

def giou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    # pred.requires_grad = True
    target = tf.convert_to_tensor(np_args[1], tf.float32)

    return [pred, target]


def executer_creator():
    return Executer(giou_loss, args_adaptor)
