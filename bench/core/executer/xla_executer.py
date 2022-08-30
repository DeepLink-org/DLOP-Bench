# Copyright (c) OpenComputeLab. All Rights Reserved.

import numpy as np
import tensorflow as tf
from .case_executer import BaseCaseExecuter
from bench.common import TFModes

tf.compat.v1.enable_eager_execution()
print(
    "Num GPUs Available: ",
    len(tf.config.experimental.list_physical_devices("GPU")),
)

TFModes_TO_XLA_ARGS = {
    TFModes.S1: {},
    TFModes.S3: {
        "experimental_compile": True
    },
}


def log_debug_info():
    """Log debug info using XLA backend.
    """
    raise NotImplementedError("no debug log for XLA.")


def tensor_type():
    """Tensor type of XLA.

    Returns:
        tf.Tensor: XLA backend tensor type.
    """
    return tf.Tensor


def trans_tensor_to_np(tensor):
    """Transform XLA tensor to numpy.

    Returns:
        numpy.ndarray: numpy format tensor value.
    """
    assert isinstance(tensor, tensor_type())
    sess = tf.Session()
    with sess.as_default():
        return tensor.eval()


def set_runtime_exec_mode(mode):
    raise NotImplementedError("can not set runtime for tensorflow.")


def get_runtime_exec_mode():
    raise NotImplementedError("can not get runtime for tensorflow.")


class XLAExecuter(BaseCaseExecuter):
    """Executer class of XLA backend.

    Args:
        core_func(Function): Sample origin execution function.
        args_adaptor(Function): Sample args adaptor function, 
            it transform numpy inputs to tensor.
    """
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)

    def prepare(self, stage_mode):
        """Prepare sample func to execute. Compile origin func
        using tf.function when it is not stage1.
        
        Args:
            stage_mode(TFModes): XLA stage.
        """
        stage_args = TFModes_TO_XLA_ARGS[stage_mode]
        self._execute_func = (self._origin_func if not stage_args else
                              tf.function(self._origin_func, **stage_args))

    def generate_args(self, case, requires_grad, np_args_generator):
        """Generate sample function execution args.
        
        Args:
            case(list): Multiple hyperparameters used to generate numpy 
                format sample function execution args.
            requires_grad(list(bool)): Requires_grad state of XLA 
                tensor format args.
            np_args_generator(Function): Transform numpy args to XLA
                tensor. 
        Returns:
            list[torch.Tensor |Any]: XLA api tensor args.
        """
        if np_args_generator is not None:
            np_args = np_args_generator(*case)
            func_args = self.adapt_args(np_args)
        else:
            func_args = self._args_adaptor(*case)
        assert len(func_args) == len(requires_grad)
        return func_args

    def clone_func_args(self, func_args):
        """Clone XLA tensor args.
        
        Args:
            func_args(list): Sample function execution args of XLA
                tensor format.
        Returns:
            list: Sample function execution args cloned of XLA
                tensor format.
        """
        assert isinstance(func_args, list)
        return [
            tf.identity(arg) if tf.is_tensor(arg) else arg for arg in func_args
        ]

    def assert_tensor_eq(self, tensor_a, tensor_b, rtol, atol):
        """Check whether the two XLA tensor values are equal.

        Args:
            tensor_a(tf.Tensor): XLA api tensor.
            tensor_b(tf.Tensor): XLA api tensor.
            rtol(float): rtol arg used in torch allclose.
            atol(float): atol arg used in torch allclose.
        """
        assert isinstance(tensor_a, tf.Tensor)
        assert isinstance(tensor_b, tf.Tensor)
        np.allclose(
            tensor_a.numpy(),
            tensor_b.numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

    def assert_correct(self, a, b, rtol, atol):
        """Check whether the two contailer or XLA tensors are equal.

        Args:
            a(tf.Tensor| list| tuple): XLA tensor or container 
                of XLA tensors.
            b(tf.Tensor| list| tuple): XLA tensor or container 
                of XLA tensors.
            rtol(float): rtol arg used in torch allclose.
            atol(float): atol arg used in torch allclose.
        """
        assert isinstance(a, (tf.Tensor, tuple, list))
        assert isinstance(b, (tf.Tensor, tuple, list))
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            assert len(a) == len(b)
            for tensor_a, tensor_b in zip(a, b):
                self.assert_tensor_eq(tensor_a, tensor_b, rtol, atol)
        elif isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
            self.assert_tensor_eq(a, b, rtol, atol)
        else:
            raise Exception("the type of `a` and `b` is different.")

    def backward(self, ret_t, func_arg):
        """Do backward for XLA tensor.
        """
        assert isinstance(ret_t, tf.Tensor)
        assert isinstance(func_arg, tf.Tensor)
        tf.gradients(ret_t, func_arg)

    def reset_grad(self, func_args, requires_grad):
        raise NotImplementedError("XLAExecuter.reset_grad: not implemented.")

    def save_timeline_start(self, case_name, stage_mode, saving_path):
        """Start record timeline.
        
        Args:
            case_name(str): Sample name.
            stage_mode(TFModes): XLA stage.
            saving_path(str): The path to save timeline.
        """
        tf.profiler.experimental.start(saving_path + case_name + "_" +
                                       stage_mode + "/")

    def save_timeline_end(self):
        """End record timeline."""
        tf.profiler.experimental.stop()
