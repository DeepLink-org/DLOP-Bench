import numpy as np
import jax.numpy as jnp
import jaxlib
from jax import jit
from jax import device_put
from .case_executer import BaseCaseExecuter
from bench.common import JAXModes


def log_debug_info():
    """Log debug info using JAX backend.
    """
    raise NotImplementedError("no debug log for jax.")


def tensor_type():
    """Tensor type of JAX.

    Returns:
        jaxlib.xla_extension.DeviceArray: JAX backend tensor type.
    """
    return jaxlib.xla_extension.DeviceArray


def trans_tensor_to_np(tensor):
    """Transform JAX tensor to numpy.

    Returns:
        numpy.ndarray: numpy format tensor value.
    """
    return jaxlib.xla_extension.DeviceArray.to_py(tensor)


def set_runtime_exec_mode(mode):
    raise NotImplementedError("can not set runtime for jax.")


def get_runtime_exec_mode():
    raise NotImplementedError("can not get runtime for jax.")

class JAXExecuter(BaseCaseExecuter):
    """Executer class of JAX backend.

    Args:
        core_func(Function): Sample origin execution function.
        args_adaptor(Function): Sample args adaptor function, 
            it transform numpy inputs to tensor.
    """
    def __init__(self, core_func, args_generator):
        super().__init__(core_func, args_generator)

    def prepare(self, stage_mode):
        """Prepare sample func to execute. Compile origin func
        using JAX jit when it is stage3.
        
        Args:
            stage_mode(JAXModes): JAX stage.
        """
        self._execute_func = (self._origin_func         \
                if not stage_mode == JAXModes.S3 else   \
                jit(self._origin_func))

    def generate_args(self, case, requires_grad, np_args_generator):
        """Generate sample function execution args.
        
        Args:
            case(list): Multiple hyperparameters used to generate numpy 
                format sample function execution args.
            requires_grad(list(bool)): Requires_grad state of JAX tensor 
                format args.
            np_args_generator(Function): Transform numpy args to JAX tensor. 
        Returns:
            list[jaxlib.xla_extension.DeviceArray|Any]: Jax tensor args.
        """
        if np_args_generator is not None:
            np_args = np_args_generator(*case)
            func_args = self.adapt_args(np_args)
        else:
            func_args = self._args_adaptor(*case)
        assert len(func_args) == len(requires_grad)
        return func_args

    def clone_func_args(self, func_args):
        """Clone Jax tensor args.
        
        Args:
            func_args(list): Sample function execution args of Jax
                tensor format.
        Returns:
            list: Sample function execution args cloned of Jax
                tensor format.
        """
        assert isinstance(func_args, list)
        return [
            jaxlib.xla_extension.DeviceArray.clone(arg) if isinstance(arg, jaxlib.xla_extension.DeviceArray) else arg for arg in func_args
        ]

    def assert_tensor_eq(self, tensor_a, tensor_b, rtol, atol):
        """Check whether the two Jax tensor values are equal.

        Args:
            tensor_a(jaxlib.xla_extension.DeviceArray): Jax tensor.
            tensor_b(jaxlib.xla_extension.DeviceArray): Jax tensor.
            rtol(float): rtol arg used in np allclose.
            atol(float): atol args used in np allclose.
        """
        assert isinstance(tensor_a, jaxlib.xla_extension.DeviceArray)
        assert isinstance(tensor_b, jaxlib.xla_extension.DeviceArray)
        np.allclose(
            jaxlib.xla_extension.DeviceArray.to_py(tensor_a),
            jaxlib.xla_extension.DeviceArray.to_py(tensor_b),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

    def assert_correct(self, a, b, rtol, atol):
        """Check whether the two contailer or JAX tensors are equal.

        Args:
            a(jaxlib.xla_extension.DeviceArray| list| tuple): Jax 
                tensor or container of JAX tensors.
            b(jaxlib.xla_extension.DeviceArray| list| tuple): Jax 
                tensor or container of JAX tensors.
            rtol(float): rtol arg used in np allclose.
            atol(float): atol args used in np allclose.
        """
        assert isinstance(a, (jaxlib.xla_extension.DeviceArray, tuple, list))
        assert isinstance(b, (jaxlib.xla_extension.DeviceArray, tuple, list))
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            assert len(a) == len(b)
            for tensor_a, tensor_b in zip(a, b):
                self.assert_tensor_eq(tensor_a, tensor_b, rtol, atol)
        elif isinstance(a, jaxlib.xla_extension.DeviceArray) and isinstance(b, jaxlib.xla_extension.DeviceArray):
            self.assert_tensor_eq(a, b, rtol, atol)
        else:
            raise Exception("the type of `a` and `b` is different.")

    def backward(self, ret_t, func_arg):
        assert isinstance(ret_t, jaxlib.xla_extension.DeviceArray)
        assert isinstance(func_arg, jaxlib.xla_extension.DeviceArray)
        pass

    def reset_grad(self, func_args, requires_grad):
        raise NotImplementedError("JAXExecuter.reset_grad: not implemented.")

    def save_timeline_start(self, case_name, stage_mode, saving_path):
        pass

    def save_timeline_end(self):
        pass
