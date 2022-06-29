import torch

from long_tail_bench.core.executer import Executer


# encode results:
# parrots: s1 ~ s5 pass
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


def encode(matched, priors):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    use_yolo_regressors = False

    if use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size(matched)

        loc = torch.cat((boxes[:, :2] - priors[:, :2],
                         torch.log(boxes[:, 2:] / priors[:, 2:])), 1)

        return loc
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def args_adaptor(np_args):
    matched = torch.from_numpy(np_args[0]).cuda()
    priors = torch.from_numpy(np_args[1]).cuda()

    return [matched, priors]


def executer_creator():
    return Executer(encode, args_adaptor)
