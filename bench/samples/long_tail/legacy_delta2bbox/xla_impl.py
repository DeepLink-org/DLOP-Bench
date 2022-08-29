# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def legacy_delta2bbox(rois,
                      deltas,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.),
                      max_shape=None,
                      wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes in the MMDet V1.x manner.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of `bbox2delta()`

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> legacy_delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.5000, 1.5000],
                [0.0000, 0.0000, 5.2183, 5.2183],
                [0.0000, 0.1321, 7.8891, 0.8679],
                [5.3967, 2.4251, 6.0033, 3.7749]])
    """
    means = tf.constant(means)
    means = tf.expand_dims(means,0)
    means = tf.tile(means, (tf.shape(deltas)[0] ,1))
    stds = tf.constant(stds)
    stds = tf.expand_dims(stds,0)
    stds = tf.tile(stds, (tf.shape(deltas)[0],1))
    #means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    #stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = tf.clip_by_value(dw, -max_ratio, max_ratio)
    dh = tf.clip_by_value(dh, -max_ratio, max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    px = tf.expand_dims(px, 1)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)
    py = tf.expand_dims(py, 1)
    pw = (rois[:, 2] - rois[:, 0] + 1.0)
    pw = tf.expand_dims(pw, 1)
    ph = (rois[:, 3] - rois[:, 1] + 1.0)
    ph = tf.expand_dims(ph, 1)
    gw = pw * tf.math.exp(dw)
    gh = ph * tf.math.exp(dh)
    #px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    #py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    #pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    #ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    #gw = pw * dw.exp()
    #gh = ph * dh.exp()
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right

    # The true legacy box coder should +- 0.5 here.
    # However, current implementation improves the performance when testing
    # the models trained in MMDetection 1.X (~0.5 bbox AP, 0.2 mask AP)
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = tf.clip_by_value(x1, 0, max_shape[1] - 1)
        y1 = tf.clip_by_value(y1, 0, max_shape[0] - 1)
        x2 = tf.clip_by_value(x2, 0, max_shape[1] - 1)
        y2 = tf.clip_by_value(y2, 0, max_shape[0] - 1)
    bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
    bboxes = tf.reshape(bboxes, deltas.shape)
    return bboxes


def args_adaptor(np_args):
    proposals = tf.convert_to_tensor(np_args[0], tf.float32)
    gt = tf.convert_to_tensor(np_args[1], tf.float32)

    return [proposals, gt, np_args[2], np_args[3]]


def executer_creator():
    return Executer(legacy_delta2bbox, args_adaptor)
