# Copyright (c) OpenComputeLab. All Rights Reserved.

from tests.test_framework.one_sample import test_sample_name, registry
from bench.common.settings import FRAMEWORK, Settings


class TestExecuter(object):
    def _prepare(self):
        case_fetcher = registry.get(test_sample_name)
        executer = case_fetcher.executer_creator()
        sample_config = case_fetcher.sample_config_getter()

        settings = Settings()
        executer.prepare(settings.frame_type_to_frame_modes[FRAMEWORK].S1)

        func_args = [
            executer.generate_args(
                case,
                sample_config.requires_grad,
                case_fetcher.np_args_generator,
            ) for case in sample_config.args_cases
        ]
        return executer, sample_config, func_args

    def test_generate_args(self):
        _, sample_config, func_args = self._prepare()
        assert len(func_args) == len(sample_config.args_cases)
        assert func_args[0][0].shape == (1, 1)

    def test_clone_func_args(self):
        executer, _, func_args = self._prepare()

        cloned_func_args = executer.clone_func_args(func_args)
        assert len(func_args) == len(cloned_func_args)
        for origin, cloned in zip(func_args, cloned_func_args):
            for a, b in zip(origin, cloned):
                assert 0 == (a != b).sum()

    def test_execute(self):
        executer, _, func_args = self._prepare()
        ret = executer.execute(func_args[0])
        assert ret[0].cpu().numpy()[0] == 3

    def test_assert_correct(self):
        executer, sample_config, func_args = self._prepare()

        cloned_func_args = executer.clone_func_args(func_args)
        reta = executer.execute(func_args[0])
        retb = executer.execute(cloned_func_args[0])
        executer.assert_correct(reta, retb, sample_config.rtol,
                                sample_config.atol)  # noqa

    def test_backward(self):
        executer, _, func_args = self._prepare()
        ret = executer.execute(func_args[0])
        executer.backward(ret)

    def test_reset_grad(self):
        executer, sample_config, func_args = self._prepare()
        executer.reset_grad(func_args[0], sample_config.requires_grad)
