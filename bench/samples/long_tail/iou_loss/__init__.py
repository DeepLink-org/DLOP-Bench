from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, ), (256, ), (512, )],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/276b7c300f72ab824c9a1087a0f7a11154e00564/mmdet/models/losses/iou_loss.py#L16",  # noqa
        tags=[SampleTag.ViewAttribute, \
              SampleTag.IfElseBranch, SampleTag.Reduce]
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    pred = gen_base(N)
    target = gen_base(N)
    return [pred, target]


register_sample(__name__, get_sample_config, gen_np_args)
