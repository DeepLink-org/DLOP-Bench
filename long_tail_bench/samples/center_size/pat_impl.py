import torch

from long_tail_bench.core.executer import Executer


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(
        (
            (boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
            boxes[:, 2:] - boxes[:, :2]),
        1)  # w, h


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    return [boxes]


def executer_creator():
    return Executer(center_size, args_adaptor)
