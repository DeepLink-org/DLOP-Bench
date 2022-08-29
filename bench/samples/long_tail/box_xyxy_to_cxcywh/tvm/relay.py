# Copyright (c) OpenComputeLab. All Rights Reserved.
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

onnx_model = onnx.load("./box_xyxy_to_cxcywh.onnx")

x = torch.randn((3000, 4), requires_grad=False)

target = "cuda"

input_name1 = "x"
shape_dict = {input_name1: x.shape}
mod, params =relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    executer = relay.build_module.create_executor(
            "graph", mod, tvm.gpu(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor(tvm.nd.array(x.astype(dtype))).numpy()
