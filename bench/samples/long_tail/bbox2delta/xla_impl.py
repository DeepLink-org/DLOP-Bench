import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    #assert proposals.size() == gt.size()

    # proposals = proposals.float()
    # gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = tf.math.log(gw / pw)
    dh = tf.math.log(gh / ph)
    deltas = tf.stack([dx, dy, dw, dh], axis=-1)

    means = np.array(means)
    means = tf.expand_dims(means, 0)
    stds = np.array(stds)
    stds = tf.expand_dims(stds, 0)

    '''means = np.array(means).unsqueeze(0)
    stds = np.array(stds).unsqueeze(0)'''

    means = tf.cast(means, dtype = tf.float32)
    stds = tf.cast(stds, dtype = tf.float32)
    deltas = deltas - means
    deltas = deltas / stds
    #deltas = deltas.sub_(means).div_(stds)

    return deltas


def args_adaptor(np_args):
    proposals = tf.convert_to_tensor(np_args[0], tf.float32)
    gt = tf.convert_to_tensor(np_args[1], tf.float32)

    means = [0, 0, 0, 0]
    stds = [1, 1, 1, 1]
    return [proposals, gt, means, stds]


def executer_creator():
    return Executer(bbox2delta, args_adaptor)
