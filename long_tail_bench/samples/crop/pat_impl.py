import torch

from long_tail_bench.core.executer import Executer


def sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size.
    Also converts from relative to absolute coordinates
    and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 *= img_size
    _x2 *= img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords
          in relative point form
    """
    with torch.no_grad():
        h, w, n = masks.size()
        boxes = boxes.clone()  # Some in-place stuff goes on here
        x1, x2 = sanitize_coordinates(boxes[:, 0],
                                      boxes[:, 2],
                                      w,
                                      padding,
                                      cast=True)
        y1, y2 = sanitize_coordinates(boxes[:, 1],
                                      boxes[:, 3],
                                      h,
                                      padding,
                                      cast=True)

        rows = torch.arange(w, device=masks.device)[None, :,
                                                    None].expand(h, w, n)
        cols = torch.arange(h, device=masks.device)[:, None,
                                                    None].expand(h, w, n)

        masks_left = rows >= x1[None, None, :]
        masks_right = rows < x2[None, None, :]
        masks_up = cols >= y1[None, None, :]
        masks_down = cols < y2[None, None, :]

        crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def args_adaptor(np_args):
    masks = torch.from_numpy(np_args[0]).cuda()
    boxes = torch.from_numpy(np_args[1]).cuda()
    padding = np_args[2]
    return [masks, boxes, padding]


def executer_creator():
    return Executer(crop, args_adaptor)
