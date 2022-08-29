import os

import torch
from torch.profiler import profile
import time

from collections import OrderedDict
from bench.common import TorchModes
from bench.common import build_pytree
from .case_executer import BaseCaseExecuter


TORCH_MODES_TO_SCRIPT_ARGS = {
    TorchModes.S1: {},
    TorchModes.S3: {"optimize": None},
}


def log_debug_info():
    raise NotImplementedError("no debug log for torch.")


def tensor_type():
    return torch.Tensor


def trans_tensor_to_np(tensor):
    assert isinstance(tensor, tensor_type())
    return tensor.detach().cpu().numpy()


def clone_tensor(value):
    if isinstance(value, tensor_type()):
        return value.clone()
    else:
        return value


def set_runtime_exec_mode(mode):
    raise NotImplementedError("can not set runtime for torch.")


def get_runtime_exec_mode():
    raise NotImplementedError("can not get runtime for torch.")


def clear_env():
    raise NotImplementedError()


class TorchAPIExecuter(BaseCaseExecuter):
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)
        self._timeline_saving_path = None

    def generate_args(self, case, requires_grad, np_args_generator):
        if np_args_generator is not None:
            np_args = np_args_generator(*case)
            func_args = self.adapt_args(np_args)
        else:
            func_args = self._args_adaptor(*case)

        assert len(func_args) == len(requires_grad)

        for arg, re_g in zip(func_args, requires_grad):
            if re_g:
                arg.requires_grad = True
        return func_args

    def clone_func_args(self, func_args):
        assert isinstance(func_args, list)
        return build_pytree(func_args, clone_tensor)

    def gen_timeline_saving_path(
        self, case_name, stage_mode, saving_path, suffix=".json"
    ):
        return os.path.join(
            saving_path,
            case_name
            + "_{}".format(str(stage_mode).replace(".", "_"))
            + "_profile"
            + suffix,
        )

    def synchronize(self):
        torch.cuda.synchronize()

    def correctness_input_types(sself):
        return (torch.Tensor, tuple, list, dict, OrderedDict)

    def correctness_dict_types(self):
        return (dict, OrderedDict)

    def assert_tensor_eq(self, tensor_a, tensor_b, rtol, atol):
        assert isinstance(tensor_a, torch.Tensor)
        assert isinstance(tensor_b, torch.Tensor)
        if tensor_a.dtype != torch.float32:
            print("tensor_a: {}", tensor_a)
            print("tensor_b: {}", tensor_b)
            assert tensor_a.equal(tensor_b)
        else:
            assert torch.allclose(
                tensor_a, tensor_b, equal_nan=True, rtol=rtol, atol=atol
            )

    def assert_correct(self, a, b, rtol, atol):
        if isinstance(a, (int, float, type(None))) and isinstance(
            b, (int, float, type(None))
        ):
            assert a == b
            return

        assert isinstance(a, self.correctness_input_types()), type(a)
        assert isinstance(b, self.correctness_input_types()), type(b)

        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            self.assert_tensor_eq(a, b, rtol, atol)
        elif isinstance(a, self.correctness_dict_types()) and isinstance(
            b, self.correctness_dict_types()
        ):
            assert len(a) == len(b)
            for k in a:
                if isinstance(a[k], torch.Tensor) and isinstance(
                    b[k], torch.Tensor
                ):
                    self.assert_tensor_eq(a[k], b[k], rtol, atol)
                elif isinstance(a[k], (int, float)) and isinstance(
                    b[k], (int, float)
                ):
                    assert a[k] == b[k]
                else:
                    self.assert_correct(a[k], b[k], rtol, atol)
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            assert len(a) == len(b), "a:{} b:{}".format(len(a), len(b))
            for element_a, element_b in zip(a, b):
                if isinstance(element_a, torch.Tensor) and isinstance(
                    element_b, torch.Tensor
                ):
                    self.assert_tensor_eq(element_a, element_b, rtol, atol)
                else:
                    self.assert_correct(element_a, element_b, rtol, atol)
        else:
            raise Exception("the type of `a` and `b` is different.")

    def backward(self, ret_t, _=None):
        assert isinstance(ret_t, torch.Tensor)
        ret_t.backward()

    def reset_grad(self, func_args, requires_grad):
        assert isinstance(func_args, list)
        assert isinstance(requires_grad, list)
        assert len(func_args) == len(requires_grad)
        for arg, re_g in zip(func_args, requires_grad):
            if re_g:
                arg.grad = None


class TorchExecuter(TorchAPIExecuter):
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)
        self._stage_args = None

    def prepare(self, stage_mode):
        self._stage_args = TORCH_MODES_TO_SCRIPT_ARGS[stage_mode]  
        self._execute_func = self._origin_func if not self._stage_args else torch.jit.script(self._origin_func, **self._stage_args)

    def save_timeline_start(self, case_name, stage_mode, saving_path):
        self._timeline_saving_path = self.gen_timeline_saving_path(
            case_name, stage_mode, saving_path
        )
        torch.autograd.profiler.profile(enable=True, use_cuda=True)

    def save_timeline_end(self):
        assert self._timeline_saving_path is not None
        prof = torch.autograd.profiler.profile(enable=False)
        prof.export_chrome_trace(self._timeline_saving_path)
    
    def get_profiler(self):
        """get torch profiler tool
        """
        return profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                with_flops=True,
        )
        
