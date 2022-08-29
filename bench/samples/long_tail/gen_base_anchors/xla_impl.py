# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def gen_base_anchors(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = tf.math.sqrt(ratios)
    w_ratios = 1 / h_ratios
    # ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
    # hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
    ws = w * w_ratios[:, None] * scales[None, :]
    ws = tf.reshape(ws, [-1])
    hs = h * h_ratios[:, None] * scales[None, :]
    hs = tf.reshape(hs, [-1])

    # base_anchors = tf.stack(
    #     [
    #         x_ctr - 0.5 * (ws - 1),
    #         y_ctr - 0.5 * (hs - 1),
    #         x_ctr + 0.5 * (ws - 1),
    #         y_ctr + 0.5 * (hs - 1),
    #     ],
    #     axis=-1,
    # ).round()
    base_anchors = tf.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        axis=-1,
    )
    base_anchors = tf.math.round(base_anchors)

    return base_anchors


def args_adaptor(np_args):
    base_size = np_args[0]
    ratios = tf.convert_to_tensor(np_args[1], tf.float32)
    scales = tf.convert_to_tensor(np_args[2], tf.float32)
    return [base_size, ratios, scales]


def executer_creator():
    return Executer(gen_base_anchors, args_adaptor)
