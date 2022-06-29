import tensorflow as tf
import numpy as np
from long_tail_bench.core.executer import Executer


def meshgrid(x, y, row_major=True):
    # xx = x.repeat(len(y))
    xx = tf.repeat(x, len(y), 0)
    # yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    yy = tf.reshape(y, [-1, 1])
    yy = tf.repeat(yy, len(x), 1)
    yy = tf.reshape(yy, [-1])
    if row_major:
        return xx, yy
    else:
        return yy, xx


def valid_flags(featmap_size, valid_size, num_base_anchors):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = tf.zeros(feat_w, dtype=tf.uint8)
    valid_y = tf.zeros(feat_h, dtype=tf.uint8)
    valid_x = tf.Variable(valid_x)
    valid_y = tf.Variable(valid_y)
    valid_x[:feat_w].assign(valid_x)
    valid_y[:feat_h].assign(valid_y)
    valid_x = tf.convert_to_tensor(valid_x)
    valid_y = tf.convert_to_tensor(valid_y)
    
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    # valid = valid[:, None].expand(
    #     valid.size(0), num_base_anchors).contiguous().reshape(-1)
    valid = valid[:, None]
    valid = tf.repeat(valid, num_base_anchors, axis=-1)
    # valid = tf.contiguous(valid)
    valid = tf.reshape(valid, [-1])
    return valid


def args_adaptor(np_args):
    featmap_size = np_args[0]
    valid_size = np_args[1]
    num_base_anchors = np_args[2]
    return [featmap_size, valid_size, num_base_anchors]


def executer_creator():
    return Executer(valid_flags, args_adaptor)
