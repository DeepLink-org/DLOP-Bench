import os
from .types import FrameType, PatModes, TorchModes, TFModes, JAXModes


def parse_env(env):
    debug = os.getenv(env)
    if debug is None:
        return False
    if debug == "1":
        return True
    elif debug == "0":
        return False
    else:
        raise Exception(
            "The environment value of {} can only be `1` or `0`.".format(env))


BENCH_DEBUG = parse_env("BENCH_DEBUG")
DEVICE_CPU = parse_env("DEVICE_CPU")
FRAMEWORK = (FrameType(os.getenv("FRAMEWORK"))
             if os.getenv("FRAMEWORK") else None)

SAMPLE_IMPL = (FrameType(os.getenv("SAMPLE_IMPL"))
               if os.getenv("SAMPLE_IMPL") else None)


class Settings(object):
    _FRAME_TYPE_TO_FRAME_MODES = {
        FrameType.Parrots: PatModes,
        FrameType.Torch: TorchModes,
        FrameType.XLA: TFModes,
        FrameType.JAX: JAXModes,
    }

    _RESULT_JSON_FILEPATH = None
    _TIME_JSON_FILEPATH = None
    _PROFILER_JSON_FILEPATH = None
    _RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "results")
    _TIME_COST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "time_results")
    _PROFILER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "profiler_results")
    _TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "..", "tests", "test_samples", "temp")

    _FRAMEWORK_COMPARE_MODE = False
    _SAMPLE_FUNC_OUTPUT_PICKLE_DIR = None
    _SAMPLE_FUNC_INPUT_PICKLE_DIR = None

    def __init__(self, framework_compare_mode=False):
        self._FRAMEWORK_COMPARE_MODE = framework_compare_mode
        self._RESULT_JSON_FILEPATH = os.path.join(self._RESULT_DIR,
                                                  FRAMEWORK.value + ".json")
        self._TIME_JSON_FILEPATH = os.path.join(self._TIME_COST_DIR,
                                                  FRAMEWORK.value + ".json")
        self._PROFILER_JSON_FILEPATH = os.path.join(self._PROFILER_DIR,
                                                  FRAMEWORK.value + ".json")
        self._SAMPLE_FUNC_OUTPUT_PICKLE_DIR = os.path.join(
            self._TEMP_DIR, "outputs_" + FRAMEWORK.value)
        self._SAMPLE_FUNC_INPUT_PICKLE_DIR = os.path.join(
            self._TEMP_DIR, "inputs")

    @property
    def frame_type_to_frame_modes(self):
        return self._FRAME_TYPE_TO_FRAME_MODES

    @property
    def result_json_filepath(self):
        return self._RESULT_JSON_FILEPATH

    @property
    def result_dir(self):
        return self._RESULT_DIR
    
    @property
    def time_json_filepath(self):
        return self._TIME_JSON_FILEPATH

    @property
    def time_dir(self):
        return self._TIME_COST_DIR
    
    @property
    def profiler_json_filepath(self):
        return self._PROFILER_JSON_FILEPATH

    @property
    def result_dir(self):
        return self._PROFILER_DIR

    @property
    def framework_compare_mode(self):
        return self._FRAMEWORK_COMPARE_MODE

    @property
    def sample_func_output_pickle_dir(self):
        return self._SAMPLE_FUNC_OUTPUT_PICKLE_DIR

    @property
    def sample_func_input_pickle_dir(self):
        return self._SAMPLE_FUNC_INPUT_PICKLE_DIR

    @property
    def temp_dir(self):
        return self._TEMP_DIR

    def set_temp_dir(self, temp_dir):
        self._TEMP_DIR = temp_dir
