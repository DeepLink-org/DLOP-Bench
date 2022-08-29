import collections
import os
import time
import pickle
import logging
import traceback
import multiprocessing as mp

from bench.common.types import PatExecMode
from .file_io.json_helper import JsonHelper
from .file_io.csv_helper import CsvHelper
from .file_io.txt_helper import TxtHelper
from bench.common import BENCH_DEBUG, DEVICE_CPU, trans_to_np,\
    unfold_custom_class, FRAMEWORK, FrameType
from bench.core.executer import tensor_type, set_runtime_exec_mode
from bench.core.registrator import custom_class_manager

import csv

import numpy as np

from bench.core.file_io import json_helper, csv_helper, txt_helper


class Engine(object):
    """Benchmark execution engine.

    Args:
        frame_type(FrameType): Backend framework type.
        settings(Settings): Benchmark settings.
        registry(Registry): Sample CaseFetcher manager.
        run_case_names(list): Sample names to run.
        run_stages(list): Stage number to execute.
        show_config(bool): Whether to show sample config.
        parrots_exec_mode(PatExecMode): Parrots runtime mode.
    """
    def __init__(
        self,
        frame_type,
        settings,
        registry,
        run_case_names,
        run_stages=None,
        show_config=False,
        parrots_exec_mode=PatExecMode.SYNC,
    ):
        self._rets = {}
        self._times = {}
        self._errors = {}
        self.content_time = {}
        self.content_profile = {}
        self._frame_type = frame_type
        self._settings = settings
        self._registry = registry
        self._run_case_names = run_case_names
        self._run_stages_idxes = (
            [st - 1 for st in run_stages] if run_stages is not None else None
        )
        self._origin_func_args = None
        self._stage_modes = settings.frame_type_to_frame_modes[frame_type]
        self._json_helper_result = JsonHelper(self._settings.result_json_filepath)
        self._csv_helper_time = CsvHelper(self._settings.time_dir)
        self._json_helper_profile = TxtHelper(self._settings.profiler_dir)
        self._show_config = show_config
        self._parrots_exec_mode = parrots_exec_mode
        

    def is_enable(self, case_name):
        """Whether the sample should run.
        """
        if not self._run_case_names:
            return True
        if case_name in self._run_case_names:
            return True
        else:
            return False

    def stage_modes_to_run(self):
        """Transform stage index to corresponding stage mode."""
        if self._run_stages_idxes is None:
            return self._stage_modes

        stage_modes = [s for s in self._stage_modes]
        return [stage_modes[idx] for idx in self._run_stages_idxes]

    def run(self):
        """Enumerate samples and execute it in a process."""
        for case_name in self._registry.key_iters():
            if not self.is_enable(case_name):
                continue
            self._origin_func_args = None
            case_fetcher = self._registry.get(case_name)
            mp.set_start_method("spawn", force=True)
            p = mp.Process(
                target=self.run_modes, args=(case_name, case_fetcher)
            )
            p.start()
            p.join()
            self.check_unknown_error(case_name, self._json_helper_result)

    def run_modes(self, case_name, case_fetcher):
        """Run all specified stage modes of a sample.

        Args:
            case_name(str): Sample name.
            case_fetcher(CaseFetcher): Get sample execution 
                related functions by it.
        """
        print(case_name, ":")
        sample_config = case_fetcher.sample_config_getter()
        if self._show_config:
            sample_config.show()
        if (
            FRAMEWORK is FrameType.Parrots
            and self._parrots_exec_mode is PatExecMode.ASYNC
        ):
            set_runtime_exec_mode("ASYNC")
        for stage_mode in self.stage_modes_to_run():
            try:
                self.run_per_mode(
                    case_name,
                    stage_mode,
                    case_fetcher.executer_creator(),
                    sample_config,
                    case_fetcher.np_args_generator,
                )
            except Exception as e:
                print(
                    "Case Name:",
                    case_name,
                    "Stage Mode:",
                    stage_mode,
                    "Error.",
                )
                self._errors[stage_mode.value] = traceback.format_exc()
                self.save_performance(
                    case_name, self._json_helper_result, sample_config
                )
                logging.exception(e)
                break
            else:
                print(
                    "Case Name:",
                    case_name,
                    "Stage Mode:",
                    stage_mode,
                    "test success!"
                )
            self.save_performance(case_name, self._json_helper_result, sample_config)

    def run_per_iter(self, executer, func_args, sample_config):
        """Run one iter using the sample executer.

        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            func_args(list): Tensor args of sample execution function.
            sample_config(SampleConfig): Sample execution config.
        Returns:
            tuple: Sample execution function results.
        """
        ret = executer.execute(func_args)
        if isinstance(ret, (int, float, str, tensor_type())):
            ret = (ret,)
        elif isinstance(ret, (dict, collections.abc.Mapping)):
            pass
        elif isinstance(ret, tuple(custom_class_manager.get_custom_classes())):
            ret = unfold_custom_class(ret)
        elif not isinstance(ret, tuple):
            ret = tuple(ret, )

        if not sample_config.backward:
            return ret

        assert len(ret) == len(sample_config.backward)
        for idx, is_backward in enumerate(sample_config.backward):
            if not is_backward:
                continue
            executer.backward(ret[idx])
            executer.reset_grad(func_args, sample_config.requires_grad)

        return ret

    def make_data(
        self,
        executer,
        sample_config,
        index=0,
        case_name=None,
        np_args_generator=None,
    ):
        """Generate sample execution function inputs.

        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            sample_config(SampleConfig): Sample execution config.
            just_one(bool): Clone one or all origin func args.
            case_name(str): Sample name.
            np_args_generator(Function): The function to generate
                numpy sample inputs.
        Returns:
            list: Sample execution function args.
        """
        if self._settings.framework_compare_mode:
            np_func_args = self.read_args_from_pickle(case_name)
            self._origin_func_args = [
                executer.adapt_args(np_func_args[index])
            ]
        else:
            self._origin_func_args = [
                executer.generate_args(
                    sample_config.args_cases[index], sample_config.requires_grad, np_args_generator
                )
            ]
        return [
            executer.clone_func_args(self._origin_func_args[0])
        ]

    def run_per_mode(
        self, case_name, stage_mode, executer, sample_config, np_args_generator
    ):
        """Run one stage mode of a sample.

        Args:
            case_name(str): Sample name.
            stage_mode(PatModes| TorchModes| TFModes| JAXModes): Running stages
                of different backend.
            executer(subclass of BaseCaseExecuter): Sample backend executer.
                sample_config(SampleConfig): Sample execution config.
            np_args_generator(Function): The function to generate
                numpy sample inputs.
        """
        if not DEVICE_CPU:
            executer.synchronize()
        executer.prepare(stage_mode)

        # correctness
        self.assert_correctness(
            executer, sample_config, stage_mode, case_name, np_args_generator
        )
        # warmup
        self.warmup(executer, sample_config, np_args_generator)

        # performance for all shapes
        samples_perf, samples_profile = self.performance_all(
            executer, sample_config, case_name, np_args_generator
        )  # noqa

        self.save_performance_all(case_name, self._csv_helper_time, self._json_helper_profile,samples_perf, samples_profile)
        
        self.performance(
            executer, sample_config, stage_mode, np_args_generator
        )  # noqa

        # timeline
        self.timeline(
            executer, sample_config, case_name, stage_mode, np_args_generator
        )
        if not DEVICE_CPU:
            executer.synchronize()

    def assert_correctness(
        self,
        executer,
        sample_config,
        stage_mode,
        case_name=None,
        np_args_generator=None,
    ):
        """Check whether the results are equal to stage1.

        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            sample_config(SampleConfig): Sample execution config.
            stage_mode(PatModes| TorchModes| TFModes| JAXModes): Running stages
                of different backend.
            case_name(str): Sample name.
            np_args_generator(Function): The function to generate
                numpy format sample inputs.
        """
        
        if stage_mode == self._stage_modes.S1 and self._settings.framework_compare_mode == False:
            return
        for idx in range(len(func_args)):
            func_args = self.make_data(
                executer,
                sample_config,
                index = idx,
                case_name=case_name,
                np_args_generator=np_args_generator,
            )  # noqa
            ret = self.run_per_iter(executer, func_args[0], sample_config)
            if stage_mode not in self._rets:
                self._rets[stage_mode] = {}
            self._rets[stage_mode][idx] = ret

            if stage_mode == self._stage_modes.S1:
                continue

            cor_args = (
                self._rets[self._stage_modes.S1][idx],
                self._rets[stage_mode][idx],
                sample_config.rtol,
                sample_config.atol,
            )
            executer.assert_correct(*cor_args)
            if BENCH_DEBUG:
                print(
                    "All Data: {} | Data {} Pass".format(
                        len(func_args), idx + 1
                    )
                )

        if self._settings.framework_compare_mode:
            rets = [ret for _, ret in self._rets[stage_mode].items()]
            self.save_res_to_pickle(case_name, trans_to_np(rets))

    def warmup(self, executer, sample_config, np_args_generator):
        """Warm up before recording sample performance.

        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            sample_config(SampleConfig): Sample execution config.
            np_args_generator(Function): The function to generate
                numpy format sample inputs.
        """
        for _ in range(sample_config.warm_up_iters):
            func_args = self.make_data(
                executer,
                sample_config,
                np_args_generator=np_args_generator,
            )
            self.run_per_iter(executer, func_args[0], sample_config)

    def performance(
        self, executer, sample_config, stage_mode, np_args_generator
    ):
        """Record sample performance.

        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            sample_config(SampleConfig): Sample execution config.
            stage_mode(PatModes| TorchModes| TFModes| JAXModes): Running stages
                of different backend.
            np_args_generator(Function): The function to generate
                numpy format sample inputs.
        """
        iters = sample_config.performance_iters
        func_args = [
            self.make_data(
                executer,
                sample_config,
                np_args_generator=np_args_generator,
            )[0]
            for _ in range(iters)
        ]

        if not DEVICE_CPU:
            executer.synchronize()
        time_start = time.time()
        for idx in range(iters):
            self.run_per_iter(executer, func_args[idx], sample_config)
        if not DEVICE_CPU:
            executer.synchronize()
        time_cost = time.time() - time_start
        self._times[stage_mode.value] = time_cost
    
    def performance_all(
        self, executer, sample_config, case_name, np_args_generator
    ):
        """Record all samples performance.
         
        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            sample_config(SampleConfig): Sample execution config.
            case_name: The name of case being performed
            np_args_generator(Function): The function to generate
                numpy format sample inputs.
                
        Returns:
            list: sample's time_cost and profile
        """
        item_num = len(sample_config.args_cases[0])
        samples_perf = {
            "item_"+str(i): []
            for i in range(item_num)
        }
        samples_perf.update({"time_cost": []})
        samples_profile = []
        

        for idx in range(len(sample_config.args_cases)):
            func_args = self.make_data(
                executer,
                sample_config,
                index=idx,
                case_name=case_name,
                np_args_generator=np_args_generator,
            )  # noqa
            
            # 判断torch
    
            with executer.get_profiler() as profiler:
                start = time.time()
                self.run_per_iter(executer, func_args[0], sample_config)
                time_cost = time.time() - start
                profiler.step()
            profile_data = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            
            if profile_data != None:
                for item_i in range(item_num):
                    samples_perf["item_"+str(item_i)].append(sample_config.args_cases[idx][item_i])
                samples_perf["time_cost"].append(str(time_cost))
                samples_profile.append(profile_data)
        if not DEVICE_CPU:
            executer.synchronize()
        return samples_perf, samples_profile

    def timeline(
        self, executer, sample_config, case_name, stage_mode, np_args_generator
    ):
        """Record sample performance.

        Args:
            executer(subclass of BaseCaseExecuter):  Sample backend executer.
            sample_config(SampleConfig): Sample execution config.
            case_name(str): Sample name.
            stage_mode(PatModes| TorchModes| TFModes| JAXModes): Running stages
                of different backend.
            np_args_generator(Function): The function to generate
                numpy format sample inputs.
        """
        if not sample_config.save_timeline:
            return

        executer.save_timeline_start(
            case_name, stage_mode, self._settings.result_dir
        )

        for _ in range(sample_config.timeline_iters):
            func_args = self.make_data(
                executer,
                sample_config,
                np_args_generator=np_args_generator,
            )
            self.run_per_iter(executer, func_args[0], sample_config)
        executer.save_timeline_end()

    def save_performance(self, case_name, json_helper, sample_config):
        """Save sample performance info to json file.
        """
        content = json_helper.read()
        source, url, tags = sample_config.show_info()
        content[case_name] = {
            "times": self._times,
            "errors": self._errors,
            "source": source,
            "url": url,
            "tags": tags,
        }
        json_helper.save(content)
    
    def save_performance_all(self, case_name, csv_helper_time, txt_helper_frofile, samples_time, samples_profile):
        
        """Save sample time_cost and profile info to csv and txt file.
        """
        item_num = len(samples_time.keys())
        length = len(samples_time["item_0"])
        csv_field_names = [
            "item_"+str(i)
            for i in range(item_num-1)
        ]
        csv_field_names.append("time_cost")
        
        time_content = samples_time
        profile_content = ""
        for i in range(length):
            dic = {       
                item: samples_time[item][i]
                for item in samples_time.keys()
            }
            profile_content = profile_content + str(dic) + "\n" + samples_profile[i]+ "\n" + "+" * 200 +"\n" + "+" * 200 +"\n"
            
        csv_helper_time.save(case_name, csv_field_names, length, time_content)
        txt_helper_frofile.save(case_name, profile_content)
            
    
    def check_unknown_error(self, case_name, json_helper):
        """Check whether there is unknown error."""
        last_mode = None
        for stage_mode in self._stage_modes:
            last_mode = stage_mode

        content = json_helper.read()
        case_content = content[case_name]
        if (
            last_mode.value not in case_content["times"]
            and len(case_content["errors"]) < 1
        ):
            case_content["errors"]["Unknown"] = "Unknown error occured."
            json_helper.save(content)

    def save_res_to_pickle(self, case_name, res):
        """Save sample function execution results to pickle file"""
        pickle_path = os.path.join(
            self._settings.sample_func_output_pickle_dir, case_name + ".pkl"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(res, f)

    def read_args_from_pickle(self, case_name):
        """Read sample function inputs from pickle file."""
        pickle_path = os.path.join(
            self._settings.sample_func_input_pickle_dir, case_name + ".pkl"
        )
        assert os.path.exists(pickle_path)
        with open(pickle_path, "rb") as f:
            return pickle.load(f)