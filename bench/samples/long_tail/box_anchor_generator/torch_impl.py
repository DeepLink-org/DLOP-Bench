# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch
import numpy as np
from bench.core.executer import Executer


class AnchorGenerator(object):
    def __init__(self):
        self._num_anchors = None
        self._num_levels = None
        self._base_anchors = None

    def build_base_anchors(self, anchor_strides):
        """build anchors over one cell"""
        raise NotImplementedError

    def get_anchors(self, featmap_shapes, device=None):
        """
        Arguments:
          - featmap_shapes: (list of tuple of (h, w, h*w*a, stride))
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    @property
    def base_anchors(self):
        return self._base_anchors

    @property
    def num_anchors(self):
        """The number of anchors per cell"""
        return self._num_anchors

    @property
    def num_levels(self):
        return self._num_levels


class BoxAnchorGenerator(AnchorGenerator):
    def __init__(self):
        self._num_anchors = None
        self._num_levels = None
        self._base_anchors = None
        self._anchor_ratios = [0.5, 1, 2]
        self._anchor_scales = 8

    def build_base_anchors(self, anchor_strides):
        self._anchor_strides = anchor_strides
        if getattr(self, "_base_anchors", None) is not None:
            return self._base_anchors
        self._num_levels = len(anchor_strides)
        self._base_anchors = []
        for idx, stride in enumerate(anchor_strides):
            anchors_over_grid = self.get_anchors_over_grid(
                self._anchor_ratios, self._anchor_scales, stride)
            self._base_anchors.append(anchors_over_grid)
        return self._base_anchors

    def get_anchors(self, featmap_shapes, device=None):
        """
        Arguments:
          - featmap_shapes: (list of tuple of (h, w, h*w*a, stride))

        Returns:
          - mlvl_anchors: (list of anchors of (h*w*a, 4))
        """
        strides = [shp[-1] for shp in featmap_shapes]
        base_anchors = self.build_base_anchors(strides)
        mlvl_anchors = []
        for (
                anchors_over_grid,
                featmap_shape,
        ) in zip(base_anchors, featmap_shapes):
            featmap_h = featmap_shape[0]
            featmap_w = featmap_shape[1]
            featmap_stride = featmap_shape[-1]
            anchors = self.get_anchors_over_plane(
                anchors_over_grid,
                featmap_h,
                featmap_w,
                featmap_stride,
                device=device,
            )
            mlvl_anchors.append(anchors)
        return mlvl_anchors

    def get_anchors_over_plane(
        self,
        anchors_over_grid,
        featmap_h,
        featmap_w,
        featmap_stride,
        dtype=torch.float32,
        device=None,
    ):
        """
        Args:
        anchors_over_grid
        """
        # [A, 4], anchors over one pixel

        anchors_over_grid = torch.from_numpy(anchors_over_grid).to(
            device=device, dtype=dtype)
        # spread anchors over each grid
        shift_x = torch.arange(
            0,
            featmap_w * featmap_stride,
            step=featmap_stride,
            dtype=dtype,
            device=device,
        )
        shift_y = torch.arange(
            0,
            featmap_h * featmap_stride,
            step=featmap_stride,
            dtype=dtype,
            device=device,
        )
        # [featmap_h, featmap_w]
        shift_y, shift_x = torch.meshgrid((shift_y, shift_x))
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

        anchors_over_grid = anchors_over_grid.reshape(1, -1, 4)
        shifts = shifts.reshape(-1, 1, 4).to(anchors_over_grid)
        anchors_overplane = anchors_over_grid + shifts
        return anchors_overplane.reshape(-1, 4)

    def get_anchors_over_grid(self, ratios, scales, stride):
        """
        generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
        """
        ratios = np.array(ratios)
        scales = np.array(scales)

        anchor = np.array([1, 1, stride, stride], dtype=np.float) - 1
        anchors = self._ratio_enum(anchor, ratios)
        anchors = np.vstack([
            self._scale_enum(anchors[i, :], scales)
            for i in range(anchors.shape[0])
        ])
        return anchors

    def _ratio_enum(self, anchor, ratios):
        """enumerate a set of anchors for each aspect ratio wrt an anchor."""
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor, scales):
        """enumerate a set of anchors for each scale wrt an anchor."""
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _whctrs(self, anchor):
        """
        return width, height, x center, and y center for an anchor (window).
        """
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """
        given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ))
        return anchors


def args_adaptor(np_args):
    featuremaps_shape = torch.from_numpy(np_args[0])
    return [featuremaps_shape]


def executer_creator():
    coder_instance = BoxAnchorGenerator()
    return Executer(coder_instance.get_anchors, args_adaptor)
