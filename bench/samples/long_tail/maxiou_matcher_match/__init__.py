# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (4000, 4), (5000, 4)],
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/platform/ParrotsDL/pytorch-object-detection/-/blob/pt/v3.1.0/pod/models/heads/utils/matcher.py#L147",  # noqa
        tags=[SampleTag.ViewAttribute, \
              SampleTag.IfElseBranch]
    )


register_sample(__name__, get_sample_config)
