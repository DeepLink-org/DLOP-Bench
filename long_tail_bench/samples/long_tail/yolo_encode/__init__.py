from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (2000, 4), (1000, 4)],
        requires_grad=[False] * 4,
        backward=[False],
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(M, N):
    bboxes = np.random.rand(M, N).astype(np.float32)
    gt_bboxes = np.random.rand(M, N).astype(np.float32)
    stride = np.random.rand(M).astype(np.float32)
    return [bboxes, gt_bboxes, stride]


register_sample(__name__, get_sample_config, gen_np_args)
