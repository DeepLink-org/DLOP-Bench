import torch
from bench.core.executer import Executer


class FcosMatcher(object):
    def __init__(self, center_sample=None, pos_radius=1):
        self.center_sample = center_sample
        self.pos_radius = pos_radius

    def match(
        self,
        points,
        gt,
        loc_ranges,
        num_points_per,
        strides=[8, 16, 32, 64, 128],
        ig=None,
    ):

        INF = 1e10
        num_gts = gt.shape[0]
        K = points.shape[0]
        gt_labels = gt[:, 4]
        xs, ys = points[:, 0], points[:, 1]
        gt_bboxes = gt[:, :4]
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] -
                                                           gt_bboxes[:, 1] + 1)

        areas = areas[None].repeat(K, 1)
        loc_ranges = loc_ranges[:, None, :]
        loc_ranges = loc_ranges.expand(K, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(K, num_gts, 4)
        gt_xs = xs[:, None].expand(K, num_gts)
        gt_ys = ys[:, None].expand(K, num_gts)

        left = gt_xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - gt_xs
        top = gt_ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - gt_ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sample:
            sample_mask = self.get_sample_region(gt_bboxes, strides,
                                                 num_points_per, gt_xs, gt_ys)
        else:
            sample_mask = bbox_targets.min(-1)[0] > 0

        max_loc_distance = bbox_targets.max(-1)[0]
        inside_loc_range = (max_loc_distance >= loc_ranges[..., 0]) & (
            max_loc_distance <= loc_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[~sample_mask] = INF

        areas[inside_loc_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(K), min_area_inds]

        # ignore
        if ig is not None:
            num_igs = ig.shape[0]
            ig_xs = xs[:, None].expand(K, num_igs)
            ig_ys = ys[:, None].expand(K, num_igs)
            ig_left = ig_xs - ig[..., 0]
            ig_right = ig[..., 2] - ig_xs
            ig_top = ig_ys - ig[..., 1]
            ig_bottom = ig[..., 3] - ig_ys
            ig_targets = torch.stack((ig_left, ig_top, ig_right, ig_bottom),
                                     -1)
            ig_inside_gt_bbox_mask = (ig_targets.min(-1)[0] > 0).max(-1)[0]
            labels[ig_inside_gt_bbox_mask] = -1
        return labels, bbox_targets

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys):
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].float().sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.bool)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * self.pos_radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0],
                                                   xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1],
                                                   ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2],
                                                   gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3],
                                                   gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask


def args_adaptor(np_args):
    points = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()
    loc_ranges = torch.from_numpy(np_args[2]).cuda()
    num_points_per = np_args[3]
    return [points, gt, loc_ranges, num_points_per]


def executer_creator():
    coder_instance = FcosMatcher()
    return Executer(coder_instance.match, args_adaptor)
