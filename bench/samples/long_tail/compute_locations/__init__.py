# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, ), (2000, ), (4000, )],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/liuyan/pytorch-object-detection/-/blob/master/pod/models/heads/utils/anchor_generator.py#L286",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.IfElseBranch]
    )


register_sample(__name__, get_sample_config)
