import os
from bench.core.engine import Engine
from bench.common.settings import FRAMEWORK, Settings
from tests.test_framework.one_sample import (
    test_sample_name,
    registry,
    count,
    reset_count,
)


class TestEngine(object):
    def _prepare(self, run_case_names=[], return_fetcher=False):
        case_fetcher = registry.get(test_sample_name)
        executer = case_fetcher.executer_creator()
        sample_config = case_fetcher.sample_config_getter()

        settings = Settings()
        executer.prepare(settings.frame_type_to_frame_modes[FRAMEWORK].S1)
        engine = Engine(FRAMEWORK, settings, registry, run_case_names)

        if return_fetcher:
            return (
                engine,
                case_fetcher,
                settings,
                case_fetcher.np_args_generator,
            )
        return (
            engine,
            executer,
            sample_config,
            settings,
            case_fetcher.np_args_generator,
        )

    def _iters_one_mode(self, sample_config):
        iters = (len(sample_config.args_cases) + sample_config.warm_up_iters +
                 sample_config.performance_iters +
                 sample_config.timeline_iters)
        return iters

    def test_is_enable(self):
        run_case_names = []
        engine, _, _, _, _ = self._prepare(run_case_names)
        assert engine.is_enable(test_sample_name)

        run_case_names = [test_sample_name]
        engine, _, _, _, _ = self._prepare(run_case_names)
        assert engine.is_enable(test_sample_name)
        assert not engine.is_enable("wrong_case_name")

    def test_run_per_iter(self):
        engine, executer, sample_config, _, np_args_generator = self._prepare()
        func_args = executer.generate_args(
            sample_config.args_cases[0],
            sample_config.requires_grad,
            np_args_generator,
        )

        ret = engine.run_per_iter(executer, func_args, sample_config)
        assert ret[0].cpu().numpy()[0] == 3

    def test_make_data(self):
        engine, executer, sample_config, _, np_args_generator = self._prepare()

        data = engine.make_data(executer,
                                sample_config,
                                np_args_generator=np_args_generator)
        assert len(data) == len(sample_config.args_cases)
        assert data[0][0].shape == sample_config.args_cases[0]

    def test_correctness(self):
        (
            engine,
            executer,
            sample_config,
            settings,
            np_args_generator,
        ) = self._prepare()
        mode = settings.frame_type_to_frame_modes[FRAMEWORK].S1
        reset_count()
        engine.assert_correctness(executer, sample_config, mode,
                                  test_sample_name, np_args_generator)
        assert count() == len(sample_config.args_cases)

    def test_warm_up(self):
        engine, executer, sample_config, _, np_args_generator = self._prepare()
        reset_count()
        engine.warmup(executer, sample_config, np_args_generator)
        assert count() == sample_config.warm_up_iters

    def test_performance(self):
        (
            engine,
            executer,
            sample_config,
            settings,
            np_args_generator,
        ) = self._prepare()
        mode = settings.frame_type_to_frame_modes[FRAMEWORK].S1
        reset_count()
        engine.performance(executer, sample_config, mode, np_args_generator)
        assert mode.value in engine._times
        assert count() == sample_config.performance_iters

    def test_timeline(self):
        (
            engine,
            executer,
            sample_config,
            settings,
            np_args_generator,
        ) = self._prepare()
        mode = settings.frame_type_to_frame_modes[FRAMEWORK].S1
        timeline_saving_path = executer.gen_timeline_saving_path(
            test_sample_name, mode, settings.result_dir, ".txt")

        if os.path.exists(timeline_saving_path):
            os.remove(timeline_saving_path)
        reset_count()
        engine.timeline(executer, sample_config, test_sample_name, mode,
                        np_args_generator)
        assert count() == sample_config.timeline_iters

    def test_run_per_mode(self):
        (
            engine,
            executer,
            sample_config,
            settings,
            np_args_generator,
        ) = self._prepare()
        mode = settings.frame_type_to_frame_modes[FRAMEWORK].S1
        reset_count()
        engine.run_per_mode(test_sample_name, mode, executer, sample_config,
                            np_args_generator)
        assert count() == self._iters_one_mode(sample_config)
