from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
# from mmcv import ops
from bench.core.executer import Executer


class BaseRoIExtractor(nn.Module, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
    """
    def __init__(self, roi_layer, out_channels, featmap_strides):
        super(BaseRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (int): The stride of input feature map w.r.t to the
                original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop("type")
        assert hasattr(ops, layer_type)  # noqa
        layer_cls = getattr(ops, layer_type)  # noqa
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @abstractmethod
    def forward(self, feats, rois, roi_scale_factor=None):
        pass


class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides)
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    # @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels,
                                       *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == "parrots":
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += (sum(x.view(-1)[0]
                                  for x in self.parameters()) * 0.0)  # noqa
        return roi_feats


def args_adaptor(np_args):
    feats = []
    for f in np_args[0]:
        torch.from_numpy(f).cuda()
        feats.append(f)

    rois = torch.from_numpy(np_args[1]).cuda()
    roi_scale_factor = 1.3
    return [feats, rois, roi_scale_factor]


def executer_creator():
    coder_instance = SingleRoIExtractor(
        roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    )
    return Executer(coder_instance.forward, args_adaptor)
