import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args, args_adaptor

onnx_model = onnx.load("./box_cxcywh_to_xyxy.onnx")


x = args_adpator(gen_np_args(200, 4))
torch_out = torch_model(x)

target = "cuda"

input_name1 = "x"
shape_dict = {input_name1: x.shape}
mod, params =relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    executer = relay.build_module.create_executor(
            "graph", mod, tvm.gpu(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor(tvm.nd.array(x.astype(dtype))).numpy()
