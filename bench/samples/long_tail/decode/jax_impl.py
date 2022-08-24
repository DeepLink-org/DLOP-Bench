import jax
import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return jnp.stack(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] + boxes[:, 2:] / 2),
        1)  # xmax, ymax


def decode(loc, priors):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    Returns: A tensor of decoded relative coordinates in point form
             form with size [num_priors, 4]
    """
    use_yolo_regressors = False
    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = jnp.stack((loc[:, :2] + priors[:, :2],
                           priors[:, 2:] * jnp.exp(loc[:, 2:])), 1)

        return point_form(boxes)
    else:
        variances = [0.1, 0.2]

        boxes = jnp.stack(
            (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
             priors[:, 2:] * jnp.exp(loc[:, 2:] * variances[1])), 1)
        # boxes[:, :2] += boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]
        return boxes


def args_adaptor(np_args):
    loc = device_put(np_args[0])
    priors = device_put(np_args[1])

    return [loc, priors]


def executer_creator():
    return Executer(decode, args_adaptor)
