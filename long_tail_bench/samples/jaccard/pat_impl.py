import torch

from long_tail_bench.core.executer import Executer


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(
                  inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(
                  inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def args_adaptor(np_args):
    box_a = torch.from_numpy(np_args[0]).cuda()
    box_b = torch.from_numpy(np_args[0]).cuda()

    return [box_a, box_b]


def executer_creator():
    return Executer(jaccard, args_adaptor)
