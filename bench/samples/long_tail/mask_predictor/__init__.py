# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(8, 16), (16, 16), (32, 16)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/platform/ParrotsDL/pytorch-object-detection/-/blob/master/pod/models/heads/htc_head/mask.py#L118",  # noqa
        tags=[SampleTag.InputAware, \
              SampleTag.ViewAttribute, SampleTag.IfElseBranch, \
              SampleTag.AdvancedIndexing, SampleTag.BuiltInDataStructure]
    )


def gen_np_args(M, N):
    def gen_base(row, column):
        data = np.random.randint(0, 1, (row, column))
        data = data.astype(np.float32)
        return data

    rois = gen_base(M, N)
    heatmap = np.array([[[7]], [[6]], [[0]], [[1]], [[2]], [[3]]] * 8)
    image_info = [
        [1, 2, 3, 4, 5, 4] * 8,
        [2, 3, 4, 5, 4, 3] * 8,
        [3, 4, 5, 1, 4, 5] * 8,
    ]
    ins = {"image_info": image_info}

    return [rois, heatmap, ins]


register_sample(__name__, get_sample_config, gen_np_args)
