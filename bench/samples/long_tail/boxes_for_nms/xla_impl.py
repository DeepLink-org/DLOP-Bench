import tensorflow as tf
import numpy as np
from long_tail_bench.core.executer import Executer


def boxes_for_nms(boxes, idxs, class_agnostic=False):
    """Boxes for NMS.
    Arguments:
    boxes (torch.Tensor): boxes in shape (N, 4).
    idxs (torch.Tensor): each index value correspond to a bbox cluster,
        and NMS will not be applied between elements of different idxs,
        shape (N, ).
    class_agnostic (bool): if true, nms is class agnostic,
        i.e. IoU thresholding happens over all boxes,
        regardless of the predicted class
    """
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = tf.reduce_max(boxes)
        offsets = idxs * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    return boxes_for_nms


def args_adaptor(np_args):
    boxes = tf.convert_to_tensor(np_args[0], tf.float32)
    idxs = tf.convert_to_tensor(np_args[1], tf.float32)
    class_agnostic = False
    return [boxes, idxs, class_agnostic]


def executer_creator():
    return Executer(boxes_for_nms, args_adaptor)
