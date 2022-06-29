import numpy as np
import math
import tensorflow as tf
from long_tail_bench.core.executer import Executer


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

def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
    target_ctry = (target[:, 1] + target[:, 3]) * 0.5
    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - tf.math.maximum(
        (target_w - 2 * tf.abs(dx)) /
        (target_w + 2 * tf.abs(dx) + eps), tf.zeros(dx.shape))
    loss_dy = 1 - tf.math.maximum(
        (target_h - 2 * tf.abs(dy)) /
        (target_h + 2 * tf.abs(dy) + eps), tf.zeros(dy.shape))
    loss_dw = 1 - tf.math.minimum(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - tf.math.minimum(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = tf.reshape(tf.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            axis=-1), [loss_dx.shape[0], -1])

    loss = tf.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)

    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    # pred.requires_grad = True

    return [pred, target]


def executer_creator():
    return Executer(bounded_iou_loss, args_adaptor)
