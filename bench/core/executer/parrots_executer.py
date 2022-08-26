import os
import torch
import parrots
from parrots.jit import pat, register_custom_class
from parrots.pytree import FrozenDict
from parrots.runtimeconfig import runtime
from parrots.log_utils import (
    set_debug_log,
    set_partial_logging,
    add_logging_part,
)
from collections import OrderedDict
from bench.common import PatModes
from bench.core.registrator import custom_class_manager
from .torch_executer import TorchAPIExecuter

PAT_MODES_TO_JIT_ARGS = {
    PatModes.S1: {},
    PatModes.S2: {
        "coderize": False,
        "fixed_shape": True,
        "optimize": False,
    },
    PatModes.S3: {
        "coderize": True,
        "fixed_shape": True,
        "optimize": False,
    },
    PatModes.S4: {
        "coderize": False,
        "fixed_shape": False,
        "optimize": False,
    },
    PatModes.S5: {
        "coderize": True,
        "fixed_shape": False,
        "optimize": False,
    },
}


def log_debug_info():
    """Log debug info using parrots backend.
    """
    runtime.exec_mode = "SYNC"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PARROTS_BT_DEPTH"] = "15"
    add_logging_part("IR")
    set_partial_logging(True)
    set_debug_log(True)


def tensor_type():
    """Tensor type of parrots.

    Returns:
        parrots.DArray: Parrots backend tensor type.
    """
    return parrots.DArray


def trans_tensor_to_np(tensor):
    """Transform parrots tensor to numpy.

    Returns:
        numpy.ndarray: numpy format tensor value.
    """
    assert isinstance(tensor, tensor_type())
    return tensor.ndarray()


def set_runtime_exec_mode(mode):
    """Set parrots runtime exec mode.

    Args:
        mode(str): `SYNC` or `ASYNC`.
    """
    assert mode in ["SYNC", "ASYNC"]
    runtime.exec_mode = mode


def get_runtime_exec_mode():
    """Get parrots runtime exec mode.
    
    Args:
        mode(str): `SYNC` or `ASYNC`.
    """
    return runtime.exec_mode


class ParrotsExecuter(TorchAPIExecuter):
    """Executer class of parrots backend.

    Args:
        core_func(Function): Sample origin execution function.
        args_adaptor(Function): Sample args adaptor function, 
            it transform numpy inputs to tensor.
    """
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)

    def prepare(self, stage_mode):
        """Prepare sample func to execute. Compile origin func
        using parrots jit when it is not stage1.
        
        Args:
            stage_mode(PatModes): Parrots stage.
        """
        stage_args = PAT_MODES_TO_JIT_ARGS[stage_mode]
        self._execute_func = (
            self._origin_func
            if not stage_args
            else pat(self._origin_func, **stage_args)
        )

    def correctness_input_types(self):
        """Tensor or container types.
        """
        return (torch.Tensor, tuple, list, dict, OrderedDict, FrozenDict)

    def correctness_dict_types(self):
        """Dict types.
        """
        return (dict, OrderedDict, FrozenDict)

    def assert_tensor_eq(self, tensor_a, tensor_b, rtol, atol):
        """Check whether the two parrots tensor values are equal.

        Args:
            tensor_a(parrots.DArray): Parrots tensor.
            tensor_b(parrots.DArray): Parrots tensor.
            rtol(float): rtol arg used in parrots allclose.
            atol(float): atol args used in parrots allclose.
        """
        assert isinstance(tensor_a, torch.Tensor), type(tensor_a)
        assert isinstance(tensor_b, torch.Tensor), type(tensor_b)
        assert tensor_a.dtype == tensor_b.dtype
        if tensor_a.dtype != torch.float32:
            equal_res = tensor_a.equal(tensor_b)
            if not equal_res:
                print("tensor_a: {}".format(tensor_a))
                print("tensor_b: {}".format(tensor_b))
                assert equal_res
        else:
            assert parrots.allclose(
                tensor_a, tensor_b, equal_nan=True, rtol=rtol, atol=atol
            )

    def save_timeline_start(self, case_name, stage_mode, saving_path):
        """Start record timeline.
        
        Args:
            case_name(str): Sample name.
            stage_mode(PatModes): Parrots stage.
            saving_path(str): The path to save timeline.
        """
        self._timeline_saving_path = self.gen_timeline_saving_path(
            case_name, stage_mode, saving_path, ".txt"
        )
        if os.path.exists(self._timeline_saving_path):
            os.remove(self._timeline_saving_path)
        parrots.runtime.profile_attrs(False)
        parrots.runtime.profile(
            enable=True, file=self._timeline_saving_path, use_scope=True
        )

    def save_timeline_end(self):
        """End record timeline."""
        parrots.runtime.profile(enable=False)

    def register_custom_class(self, obj):
        """Register custom class.

        Args:
            obj(Any): User custom class used in 
                sample execution function inputs or outputs.
        Returns:
            ParrotsExecuter: self.
        """
        assert obj is not None
        register_custom_class(obj)
        custom_class_manager.register_class(obj)
        return self
