import os
import pickle
import shutil
import argparse
from enum import Enum
from posix import listdir
from long_tail_bench.core.engine import Engine
from long_tail_bench.common.settings import FRAMEWORK, Settings
from long_tail_bench.core import registry

# Register all cases
from long_tail_bench import samples  # noqa


class TaskType(Enum):
    Test_Parrots = "test_parrots"
    Test_Torch = "test_torch"
    Test_XLA = "test_xla"
    Compare_Res = "compare_results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        help="different stage of samples testing, such as test_parrots, \
                test_torch, test_xla, compare_results",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--cases",
        help="the cases to run, split by `,`",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.cases is not None:
        args.cases = str(args.cases).split(",")
    else:
        args.cases = []
        for case_name in registry.key_iters():
            case_fetcher = registry.get(case_name)
            if case_fetcher.np_args_generator is None:
                continue
            else:
                args.cases.append(case_name)
    return args


class SampleChecker(object):
    def __init__(self):
        self.settings = Settings(framework_compare_mode=True)

    def gen_numpy_inputs(self, run_case_names):
        if os.path.exists(self.settings._TEMP_DIR):
            shutil.rmtree(self.settings._TEMP_DIR)
        os.mkdir(self.settings._TEMP_DIR)
        os.mkdir(self.settings._SAMPLE_FUNC_INPUT_PICKLE_DIR)
        for case_name in run_case_names:
            case_fetcher = registry.get(case_name)
            if case_fetcher.np_args_generator is None:
                continue

            sample_config = case_fetcher.sample_config_getter()
            sample_args = []
            for arg_case in sample_config.args_cases:
                func_args = case_fetcher.np_args_generator(*arg_case)
                sample_args.append(func_args)
            if len(sample_args) <= 0:
                continue
            # save sample args to pickle
            pkl_path = os.path.join(
                self.settings.sample_func_input_pickle_dir,
                case_name + ".pkl",
            )
            with open(pkl_path, "wb") as f:
                pickle.dump(sample_args, f)

    def gen_framework_res(self, run_case_names):
        if not os.path.exists(self.settings._SAMPLE_FUNC_OUTPUT_PICKLE_DIR):
            os.mkdir(self.settings._SAMPLE_FUNC_OUTPUT_PICKLE_DIR)

        benchmark_engine = Engine(FRAMEWORK,
                                  self.settings,
                                  registry,
                                  run_case_names,
                                  run_stages=[1])
        benchmark_engine.run()

    def compare_results(self, run_case_names):
        dirs = listdir(self.settings.temp_dir)
        dir_names = [dir_name for dir_name in dirs if "outputs" in dir_name]

        for case_name in run_case_names:
            case_cache = dict.fromkeys(dir_names, None)
            for dir_name in dirs:
                if "outputs" not in dir_name:
                    continue
                pkl_path = os.path.join(self.settings.temp_dir, dir_name,
                                        case_name + ".pkl")
                if not os.path.exists(pkl_path):
                    case_cache = None
                else:
                    with open(pkl_path, "rb") as f:
                        case_cache[dir_name] = pickle.load(f)
            assert len(case_cache) > 1
            last_dir_name = None
            passed = True
            for dir_name in dir_names:
                if last_dir_name is None:
                    last_dir_name = dir_name
                    continue

                for a, b in zip(case_cache[dir_name],
                                case_cache[last_dir_name]):
                    if isinstance(a, tuple) and isinstance(b, tuple):
                        for ret_a, ret_b in zip(a, b):
                            if ret_a.all() != ret_b.all():
                                passed = False
                                print("{}: {} is not equal with {}.".format(
                                    case_name, last_dir_name, dir_name))
                    else:
                        if a != b:
                            passed = False
                            print("{}: {} is not equal with {}.".format(
                                case_name, last_dir_name, dir_name))
            if passed:
                print("{} passed.".format(case_name))


if __name__ == "__main__":
    args = parse_args()
    task = TaskType(args.task)
    test_samples = SampleChecker()
    if task == TaskType.Test_Parrots:
        test_samples.gen_numpy_inputs(args.cases)
        test_samples.gen_framework_res(args.cases)
    elif task == TaskType.Compare_Res:
        test_samples.compare_results(args.cases)
    else:
        test_samples.gen_framework_res(args.cases)
