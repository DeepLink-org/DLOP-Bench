# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def tblr2bboxes(priors,
                tblr,
                normalizer=4.0,
                normalize_by_wh=True,
                max_shape=None):
    """Decode tblr outputs to prediction boxes.

    The process includes 3 steps: 1) De-normalize tblr coordinates by
    multiplying it with `normalizer`; 2) De-normalize tblr coordinates by the
    prior bbox width and height if `normalize_by_wh` is `True`; 3) Convert
    tblr (top, bottom, left, right) pair relative to the center of priors back
    to (xmin, ymin, xmax, ymax) coordinate.

    Args:
        priors (Tensor): Prior boxes in point form (x0, y0, x1, y1)
          Shape: (n,4).
        tblr (Tensor): Coords of network output in tblr form
          Shape: (n, 4).
        normalizer (Sequence[float] | float): Normalization parameter of
          encoded boxes. By list, it represents the normalization factors at
          tblr dims. By float, it is the unified normalization factor at all
          dims. Default: 4.0
        normalize_by_wh (bool): Whether the tblr coordinates have been
          normalized by the side length (wh) of prior bboxes.
        max_shape (tuple, optional): Shape of the image. Decoded bboxes
          exceeding which will be clamped.

    Return:
        encoded boxes (Tensor), Shape: (n, 4)
    """
    if not isinstance(normalizer, float):
        normalizer = tf.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    #assert priors.size(0) == tblr.size(0)
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = tf.split(wh, 2, axis=1)
        '''loc_decode[:, :2] *= h  # tb
        loc_decode[:, 2:] *= w  # lr'''
        '''loc_decode.at[:, :2].mul(h)
        loc_decode.at[:, 2:].mul(w)'''
        locx = loc_decode[:, :2]
        locy = loc_decode[:, 2:]
        locx *= h
        locy *= w
        locx2 = tf.Variable(loc_decode)
        locy2 = tf.Variable(loc_decode)
        locx2[:, :2].assign(locx)
        locy2[:, 2:].assign(locy)
    top, bottom, left, right = tf.split(loc_decode, 4, axis=1)
    xmin = tf.expand_dims(prior_centers[:, 0], 1) - left
    xmax = tf.expand_dims(prior_centers[:, 0], 1) + right
    ymin = tf.expand_dims(prior_centers[:, 1], 1) - top
    ymax = tf.expand_dims(prior_centers[:, 1], 1) + bottom
    boxes = tf.stack((xmin, ymin, xmax, ymax), axis=1)
    if max_shape is not None:
        boxes[:, 0].clamp(min=0, max=max_shape[1])
        boxes[:, 1].clamp(min=0, max=max_shape[0])
        boxes[:, 2].clamp(min=0, max=max_shape[1])
        boxes[:, 3].clamp(min=0, max=max_shape[0])
    return boxes


def args_adaptor(np_args):
    priors = tf.convert_to_tensor(np_args[0], tf.float32)
    tblr = tf.convert_to_tensor(np_args[1], tf.float32)

    return [priors, tblr]


def executer_creator():
    return Executer(tblr2bboxes, args_adaptor)
