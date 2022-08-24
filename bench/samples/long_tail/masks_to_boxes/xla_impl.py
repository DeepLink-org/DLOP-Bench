import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    # if tf.size(masks) == 0:
    #     return tf.zeros((0, 4))

    h, w = masks.shape[-2:]     
    y = tf.range(0, h, dtype=tf.float32)
    x = tf.range(0, w, dtype=tf.float32)
    y, x = tf.meshgrid(y, x)
    x_mask = (masks * tf.expand_dims(x, axis=0))    
    x_max = tf.math.reduce_max(tf.reshape(x_mask, (x_mask.shape[0], -1)), axis=-1)
    x_min = tf.math.reduce_min(tf.reshape(tf.where(~tf.cast(masks, tf.bool), 1e8, x_mask), (x_mask.shape[0], -1)), axis=-1)

    y_mask = (masks * tf.expand_dims(y, 0))
    y_max = tf.math.reduce_max(tf.reshape(y_mask, (y_mask.shape[0], -1)), axis=-1)
    y_min = tf.math.reduce_min(tf.reshape(tf.where(~tf.cast(masks, tf.bool), 1e8, y_mask), (y_mask.shape[0], -1)), axis=-1)

    return tf.stack([x_min, y_min, x_max, y_max], 1)


def args_adaptor(np_args):
    masks = tf.convert_to_tensor(np_args[0], tf.float32)

    return [masks]


def executer_creator():
    return Executer(masks_to_boxes, args_adaptor)
