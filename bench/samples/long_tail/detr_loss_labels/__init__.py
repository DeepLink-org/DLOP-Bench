from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)


def get_sample_config():
    return SampleConfig(
        args_cases=[(100, ), (200, )],
        requires_grad=[False] * 4,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/detr.py#L108",  # noqa
        tags=[SampleTag.BuiltInDataStructure, SampleTag.ForLoop,
              SampleTag.IfElseBranch, SampleTag.AdvancedIndexing]
    )


register_sample(__name__, get_sample_config)
