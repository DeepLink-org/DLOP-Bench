# Copyright (c) OpenComputeLab. All Rights Reserved.

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
    TorchModes.S2: {"optimize": None},
}


def log_debug_info():
    """Log debug info using torch backend.
    """
    raise NotImplementedError("no debug log for torch.")


def tensor_type():
    """Tensor type of torch.

    Returns:
        torch.Tensor: Torch backend tensor type.
    """
    return torch.Tensor


def trans_tensor_to_np(tensor):
    """Transform torch tensor to numpy.

    Returns:
        numpy.ndarray: numpy format tensor value.
    """
    assert isinstance(tensor, tensor_type())
    return tensor.detach().cpu().numpy()


def clone_tensor(value):
    """Clone torch tensor.

    Args:
        value(torch.Tensor| Any): Torch tensor.
    Returns:
        torch.Tensor| Any: Torch tensor cloned.
    """
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
    """Executer class using torch api.

    Args:
        core_func(Function): Sample origin execution function.
        args_adaptor(Function): Sample args adaptor function, 
            it transform numpy inputs to tensor.
    """
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)
        self._timeline_saving_path = None

    def generate_args(self, case, requires_grad, np_args_generator):
        """Generate sample function execution args.
        
        Args:
            case(list): Multiple hyperparameters used to generate numpy 
                format sample function execution args.
            requires_grad(list(bool)): Requires_grad state of torch api 
                tensor format args.
            np_args_generator(Function): Transform numpy args to torch 
                api tensor. 
        Returns:
            list[torch.Tensor |Any]: Torch api tensor args.
        """
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
        """Clone Torch api tensor args.
        
        Args:
            func_args(list): Sample function execution args of torch
                api tensor format.
        Returns:
            list: Sample function execution args cloned of torch
                api tensor format.
        """
        assert isinstance(func_args, list)
        return build_pytree(func_args, clone_tensor)

    def gen_timeline_saving_path(
        self, case_name, stage_mode, saving_path, suffix=".json"
    ):
        """Generate timeline saving path.

        Args:
            case_name(str): Sample name.
            stage_mode(TorchModes):Benchmark execution stage.
            saving_path(str): The path to save timeline.
        Returns:
            str: Timeline saving path.
        """
        return os.path.join(
            saving_path,
            case_name
            + "_{}".format(str(stage_mode).replace(".", "_"))
            + "_profile"
            + suffix,
        )

    def synchronize(self):
        """Do synchronize using torch api.
        """
        torch.cuda.synchronize()

    def correctness_input_types(sself):
        """Tensor or container types.
        """
        return (torch.Tensor, tuple, list, dict, OrderedDict)

    def correctness_dict_types(self):
        """Dict types.
        """
        return (dict, OrderedDict)

    def assert_tensor_eq(self, tensor_a, tensor_b, rtol, atol):
        """Check whether the two torch api tensor values are equal.

        Args:
            tensor_a(torch.Tensor): Torch api tensor.
            tensor_b(torch.Tensor): Torch api tensor.
            rtol(float): rtol arg used in torch allclose.
            atol(float): atol arg used in torch allclose.
        """
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
        """Check whether the two contailer or torch api tensors are equal.

        Args:
            a(torch.Tensor| list| tuple): Torch api tensor or container 
                of torch api tensors.
            b(torch.Tensor| list| tuple): Torch api tensor or container 
                of torch api tensors.
            rtol(float): rtol arg used in torch allclose.
            atol(float): atol args used in torch allclose.
        """
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
        """Do backward for torch api tensor.
        """
        assert isinstance(ret_t, torch.Tensor)
        ret_t.backward()

    def reset_grad(self, func_args, requires_grad):
        """Reset grad of torch api tensors.

        Args:
            func_args(list): Sample execution function input args.
            requires_grad(list): Requires_grad state of the args.
        """
        assert isinstance(func_args, list)
        assert isinstance(requires_grad, list)
        assert len(func_args) == len(requires_grad)
        for arg, re_g in zip(func_args, requires_grad):
            if re_g:
                arg.grad = None


class TorchExecuter(TorchAPIExecuter):
    """Executer class of Torch backend.

    Args:
        core_func(Function): Sample origin execution function.
        args_adaptor(Function): Sample args adaptor function, 
            it transform numpy inputs to tensor.
    """
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)
        self._stage_args = None

    def prepare(self, stage_mode):
        """Prepare sample func to execute. Compile origin func
        using torch jit when it is not stage1.
        
        Args:
            stage_mode(TorchModes): Torch stage.
        """
        stage_args = TORCH_MODES_TO_SCRIPT_ARGS[stage_mode]
        self._execute_func = (
            self._origin_func
            if not stage_args
            else torch.jit.script(self._origin_func, **stage_args)
        )

    def save_timeline_start(self, case_name, stage_mode, saving_path):
        """Start record timeline.
        
        Args:
            case_name(str): Sample name.
            stage_mode(TorchModes): Torch stage.
            saving_path(str): The path to save timeline.
        """
        self._timeline_saving_path = self.gen_timeline_saving_path(
            case_name, stage_mode, saving_path
        )
        torch.autograd.profiler.profile(enable=True, use_cuda=True)

    def save_timeline_end(self):
        """End record timeline."""
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
        
