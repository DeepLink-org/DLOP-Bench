import numpy as np
import tensorflow as tf
from .case_executer import BaseCaseExecuter
from long_tail_bench.common import TFModes

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
    raise NotImplementedError("no debug log for xla.")


def tensor_type():
    return tf.Tensor


def trans_tensor_to_np(tensor):
    assert isinstance(tensor, tensor_type())
    sess = tf.Session()
    with sess.as_default():
        return tensor.eval()


def set_runtime_exec_mode(mode):
    raise NotImplementedError("can not set runtime for tensorflow.")


def get_runtime_exec_mode():
    raise NotImplementedError("can not get runtime for tensorflow.")


class XLAExecuter(BaseCaseExecuter):
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)

    def prepare(self, stage_mode):
        stage_args = TFModes_TO_XLA_ARGS[stage_mode]
        self._execute_func = (self._origin_func if not stage_args else
                              tf.function(self._origin_func, **stage_args))

    def generate_args(self, case, requires_grad, np_args_generator):
        if np_args_generator is not None:
            np_args = np_args_generator(*case)
            func_args = self.adapt_args(np_args)
        else:
            func_args = self._args_adaptor(*case)
        assert len(func_args) == len(requires_grad)
        return func_args

    def clone_func_args(self, func_args):
        assert isinstance(func_args, list)
        return [
            tf.identity(arg) if tf.is_tensor(arg) else arg for arg in func_args
        ]

    def assert_tensor_eq(self, tensor_a, tensor_b, rtol, atol):
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
        assert isinstance(ret_t, tf.Tensor)
        assert isinstance(func_arg, tf.Tensor)
        tf.gradients(ret_t, func_arg)

    def reset_grad(self, func_args, requires_grad):
        raise NotImplementedError("XLAExecuter.reset_grad: not implemented.")

    def save_timeline_start(self, case_name, stage_mode, saving_path):
        tf.profiler.experimental.start(saving_path + case_name + "_" +
                                       stage_mode + "/")

    def save_timeline_end(self):
        tf.profiler.experimental.stop()
