# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    auto_import,
    SampleSource,
    SampleTag
)
from bench.core import CaseFetcher, registry
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(300, 8), (600, 16), (800, 32)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMEDIT,
        url="https://github.com/open-mmlab/mmediting/blob/23213c839ff2d1907a80d6ea29f13c32a24bb8ef/mmedit/models/losses/gan_loss.py#L75",  # noqa
        tags=[SampleTag.IfElseBranch, SampleTag.Reduce]
    )


def gen_np_args(M, N):
    inputs = np.random.rand(M, N)
    inputs = inputs.astype(np.float32)
    return [inputs]


backend = auto_import(__name__)
if backend is not None:
    registry.register(
        "ganloss_vanilla",
        CaseFetcher(backend.vanilla_executer_creator, get_sample_config,
                    gen_np_args),
    )
    registry.register(
        "ganloss_lsgan",
        CaseFetcher(backend.lsgan_executer_creator, get_sample_config,
                    gen_np_args),
    )

    registry.register(
        "ganloss_wgan",
        CaseFetcher(backend.wgan_executer_creator, get_sample_config,
                    gen_np_args),
    )

    registry.register(
        "ganloss_hinge",
        CaseFetcher(backend.hinge_executer_creator, get_sample_config,
                    gen_np_args),
    )
