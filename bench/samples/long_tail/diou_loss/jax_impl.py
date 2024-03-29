# Copyright (c) OpenComputeLab. All Rights Reserved.

from jax import numpy as np
from jax import device_put, grad, value_and_grad
from bench.core.executer import Executer


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (ndarray): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(ndarray): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = numpy.ndarray([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = numpy.ndarray([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        array([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = numpy.ndarray([])
        >>> nonempty = numpy.ndarray([
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

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = np.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clip(a_min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = np.minimum(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = np.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    else:
        lt = np.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = np.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clip(a_min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou' or mode == 'giou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]
            
        if mode == 'giou':
            enclosed_lt = np.minimum(bboxes1[:, :, None, :2],
                                    bboxes2[:, None, :, :2])
            enclosed_rb = np.maximum(bboxes1[:, :, None, 2:],
                                    bboxes2[:, None, :, 2:])

    eps = np.array([eps])
    union = np.maximum(union, eps)
    ious = overlap / union

    if mode in ['iou', 'iof']:
        return ious
    
    # calculate gious
    enclose_wh = np.clip((enclosed_rb - enclosed_lt), 0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    enclose_area = np.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def diou_loss(pred, target, eps=1e-6):
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
    lt = np.maximum(pred[:, :2], target[:, :2])
    rb = np.minimum(pred[:, 2:], target[:, 2:])
    wh = np.clip((rb - lt), 0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = np.minimum(pred[:, :2], target[:, :2])
    enclose_x2y2 = np.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = np.clip((enclose_x2y2 - enclose_x1y1), 0)

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

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


def args_adaptor(np_args):
    pred = device_put(np_args[0])
    target = device_put(np_args[1])
    return [pred, target]


def executer_creator():
    return Executer(diou_loss, args_adaptor)
