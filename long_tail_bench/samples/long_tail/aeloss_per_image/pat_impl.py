import torch
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


def ae_loss_per_image(tl_preds, br_preds, match):
    """Associative Embedding Loss in one image.

    Associative Embedding Loss including two parts: pull loss and push loss.
    Pull loss makes embedding vectors from same object closer to each other.
    Push loss distinguish embedding vector from different objects, and makes
        the gap between them is large enough.

    During computing, usually there are 3 cases:
        - no object in image: both pull loss and push loss will be 0.
        - one object in image: push loss will be 0 and pull loss is computed
            by the two corner of the only object.
        - more than one objects in image: pull loss is computed by corner pairs
            from each object, push loss is computed by each object with all
            other objects. We use confusion matrix with 0 in diagonal to
            compute the push loss.

    Args:
        tl_preds (tensor): Embedding feature map of left-top corner.
        br_preds (tensor): Embedding feature map of bottim-right corner.
        match (list): Downsampled coordinates pair of each ground truth box.
    """

    tl_list, br_list, me_list = [], [], []
    if len(match) == 0:  # no object in image
        pull_loss = tl_preds.sum() * 0.0
        push_loss = tl_preds.sum() * 0.0
    else:
        for m in match:
            [tl_y, tl_x], [br_y, br_x] = m
            tl_e = tl_preds[:, tl_y, tl_x].view(-1, 1)
            br_e = br_preds[:, br_y, br_x].view(-1, 1)
            tl_list.append(tl_e)
            br_list.append(br_e)
            me_list.append((tl_e + br_e) / 2.0)

        tl_list = torch.cat(tl_list)
        br_list = torch.cat(br_list)
        me_list = torch.cat(me_list)

        assert tl_list.size() == br_list.size()

        # N is object number in image, M is dimension of embedding vector
        N, M = tl_list.size()

        pull_loss = (tl_list - me_list).pow(2) + (br_list - me_list).pow(2)
        pull_loss = pull_loss.sum() / N

        margin = 1  # exp setting of CornerNet, details in section 3.3 of paper

        # confusion matrix of push loss
        conf_mat = me_list.expand((N, N, M)).permute(1, 0, 2) - me_list
        # conf_weight = 1 - torch.eye(N).type_as(me_list)
        # parrots do not support coderize cpu op
        conf_weight = 1 - torch.eye(N, device="cuda").type_as(me_list)
        conf_mat = conf_weight * (margin - conf_mat.sum(-1).abs())

        if N > 1:  # more than one object in current image
            push_loss = F.relu(conf_mat).sum() / (N * (N - 1))
        else:
            push_loss = tl_preds.sum() * 0.0

    return pull_loss, push_loss


def args_adaptor(np_args):
    tl_preds = torch.from_numpy(np_args[0]).cuda()
    br_preds = torch.from_numpy(np_args[1]).cuda()
    match = np_args[2]
    return [tl_preds, br_preds, match]


def executer_creator():
    return Executer(ae_loss_per_image, args_adaptor)
