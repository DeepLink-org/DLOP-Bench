from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(32, 4), (64, 4), (16, 4)],
        requires_grad=[False] * 5,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/losses/pisa_loss.py#L123",  # noqa
        tags=[SampleTag.InputAware,
              SampleTag.ViewAttribute, SampleTag.IfElseBranch,
              SampleTag.BuiltInDataStructure, SampleTag.Reduce,
              SampleTag.AdvancedIndexing]
    )


def gen_np_args(M, N):
    num_class = 80
    cls_score = np.random.rand(M, num_class)
    labels = np.random.randint(num_class, size=M)
    bbox_pred = np.random.rand(M, N)
    bbox_targets = np.random.rand(M, N)

    return [cls_score, labels, bbox_pred, bbox_targets]


register_sample(__name__, get_sample_config, gen_np_args)
