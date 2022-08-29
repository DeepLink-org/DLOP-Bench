import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args, args_adaptor

onnx_model = onnx.load("./masks_to_boxes.onnx")


masks = args_adpator(gen_np_args(300, 128, 6))
torch_out = torch_model(masks)

target = "cuda"

input_name1 = "masks"
shape_dict = {input_name1: masks.shape}
mod, params =relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    executer = relay.build_module.create_executor(
            "graph", mod, tvm.gpu(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor(tvm.nd.array(masks.astype(dtype))).numpy()
