from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(5, 128), (5, 64), (5, 256)],
        requires_grad=[False] * 4,
        backward=[False, False],
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/platform/ParrotsDL/pytorch-object-detection/-/blob/master/pod/models/heads/fcos_head/fcos.py#L156",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.IfElseBranch,
              SampleTag.Reduce, SampleTag.AdvancedIndexing,
              SampleTag.ForLoop]
    )


def gen_np_args(M, N):
    shape = (M, N)
    points = np.random.randint(0, 2, (shape))
    points = points.astype(np.float32)
    gt = np.random.randint(0, 2, (shape))
    gt = gt.astype(np.float32)
    loc_ranges = np.array([[-1, 64], [64, 128], [128, 256], [256, 512],
                           [512, 100000]])
    num_points_per = 6
    return [points, gt, loc_ranges, num_points_per]


register_sample(__name__, get_sample_config, gen_np_args)
