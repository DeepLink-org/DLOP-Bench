import jax.numpy as jnp
import jax
from jax import device_put
from jax import numpy as np
from long_tail_bench.core.executer import Executer



def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = jax.numpy.clip(rb - lt, 1.0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # parrots jit can not trace, so we delete it
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = np.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = jax.numpy.clip(rb - lt, 1.0)  # [N,M,2]  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def args_adaptor(np_args):
    boxes1 = device_put(np_args[0])
    boxes2 = device_put(np_args[1])
    return [boxes1, boxes2]


def executer_creator():
    return Executer(generalized_box_iou, args_adaptor)
