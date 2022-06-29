import tensorflow as tf
import numpy as np
from long_tail_bench.core.executer import Executer


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    # if isinstance(deltas, torch.cuda.HalfTensor):
    #     deltas = deltas.float()
    means = tf.constant(means)
    means = tf.expand_dims(means,0)
    means = tf.tile(means, (tf.shape(deltas)[0] ,1))
    stds = tf.constant(stds)
    stds = tf.expand_dims(stds,0)
    stds = tf.tile(stds, (tf.shape(deltas)[0],1))
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
    # gx = tf.math.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    # gy = tf.math.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gx = px + pw * dx
    gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = tf.clip_by_value(x1, 0, max_shape[1] - 1)
        y1 = tf.clip_by_value(y1, 0, max_shape[0] - 1)
        x2 = tf.clip_by_value(x2, 0, max_shape[1] - 1)
        y2 = tf.clip_by_value(y2, 0, max_shape[0] - 1)
    bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
    bboxes = tf.reshape(bboxes, deltas.shape)
    return bboxes


def args_adaptor(np_args):
    rois = tf.convert_to_tensor(np_args[0], tf.float32)
    deltas = tf.convert_to_tensor(np_args[1], tf.float32)

    means = [0.0, 0.0, 0.0, 0.0]
    stds = [1.0, 1.0, 1.0, 1.0]
    max_shape = tf.convert_to_tensor(np_args[4], tf.float32)

    return [rois, deltas, means, stds, max_shape]


def executer_creator():
    return Executer(delta2bbox, args_adaptor)
