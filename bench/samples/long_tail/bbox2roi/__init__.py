from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)


def get_sample_config():
    return SampleConfig(
        args_cases=[(32,), (64,), (128,)],
        requires_grad=[False],
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.ForLoop, SampleTag.IfElseBranch],
    )


register_sample(__name__, get_sample_config)
