import tensorflow as tf
import numpy as np
from bench.core.executer import Executer

def bboxes2tblr(priors, gts, normalizer=4.0, normalize_by_wh=True):
    """Encode ground truth boxes to tblr coordinate.

    It first convert the gt coordinate to tblr format,
     (top, bottom, left, right), relative to prior box centers.
     The tblr coordinate may be normalized by the side length of prior bboxes
     if `normalize_by_wh` is specified as True, and it is then normalized by
     the `normalizer` factor.

    Args:
        priors (Tensor): Prior boxes in point form
            Shape: (num_proposals,4).
        gts (Tensor): Coords of ground truth for each prior in point-form
            Shape: (num_proposals, 4).
        normalizer (Sequence[float] | float): normalization parameter of
            encoded boxes. If it is a list, it has to have length = 4.
            Default: 4.0
        normalize_by_wh (bool): Whether to normalize tblr coordinate by the
            side length (wh) of prior bboxes.

    Return:
        encoded boxes (Tensor), Shape: (num_proposals, 4)
    """

    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    xmin, ymin, xmax, ymax = tf.split(gts, 4, axis=1)
    top = prior_centers[:, 1]
    top = tf.expand_dims(top, 1)- ymin
    bottom = prior_centers[:, 1]
    bottom = ymax - tf.expand_dims(bottom, 1)
    left = prior_centers[:, 0]
    left = tf.expand_dims(left, 1) - xmin
    right = prior_centers[:, 0]
    right = xmax - tf.expand_dims(right, 1)
    loc = tf.stack((top, bottom, left, right), axis=1)
    if normalize_by_wh:
        # Normalize tblr by anchor width and height
        wh = priors[:, 2:4] - priors[:, 0:2]
        wh = tf.expand_dims(wh, 2)
        w, h = tf.split(wh, 2, axis=1)
        loc[:, :2] /= h  # tb is normalized by h
        loc[:, 2:] /= w  # lr is normalized by w
    # Normalize tblr by the given normalization factor
    return loc / normalizer


def args_adaptor(np_args):
    priors = tf.convert_to_tensor(np_args[0], tf.float32)
    gts = tf.convert_to_tensor(np_args[1], tf.float32)
    return [priors, gts]


def executer_creator():
    return Executer(bboxes2tblr, args_adaptor)
