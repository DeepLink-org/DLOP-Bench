import torch
from long_tail_bench.core.executer import Executer


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def args_generator(N):
    img_num = 8
    bbox_list = []
    for _ in range(img_num):
        boxes = torch.randn((N, 4), dtype=torch.float32, device="cuda")
        bbox_list.append(boxes)
    return [bbox_list]


def executer_creator():
    return Executer(bbox2roi, args_generator)
