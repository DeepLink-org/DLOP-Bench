import torch
from long_tail_bench.core.executer import Executer
from long_tail_bench.samples.encode.pat_impl import encode


def get_target_single_tensor(pos_bboxes, neg_bboxes, pos_gt_bboxes,
                             pos_gt_labels, labels, label_weights,
                             bbox_targets, bbox_weights, cfg):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    reg_decoded_bbox = False
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg['pos_weight'] <= 0 else cfg['pos_weight']
        label_weights[:num_pos] = pos_weight
        if not reg_decoded_bbox:
            pos_bbox_targets = encode(pos_bboxes, pos_gt_bboxes)
        else:
            pos_bbox_targets = pos_gt_bboxes
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights


def args_adaptor(np_args):
    pos_bboxes = torch.from_numpy(np_args[0]).cuda()
    neg_bboxes = torch.from_numpy(np_args[1]).cuda()
    pos_gt_bboxes = torch.from_numpy(np_args[2]).cuda()
    pos_gt_labels = torch.from_numpy(np_args[3]).cuda()
    labels = torch.from_numpy(np_args[4]).cuda()
    label_weights = torch.from_numpy(np_args[5]).cuda()
    bbox_targets = torch.from_numpy(np_args[6]).cuda()
    bbox_weights = torch.from_numpy(np_args[7]).cuda()
    cfg = {'pos_weight': -1}
    return [
        pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, labels,
        label_weights, bbox_targets, bbox_weights, cfg
    ]


def executer_creator():
    return Executer(get_target_single_tensor, args_adaptor)
