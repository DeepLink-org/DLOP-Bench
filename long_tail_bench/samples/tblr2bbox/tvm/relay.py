import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args

onnx_model = onnx.load("./tblr2bboxes.onnx")


priors, tblr = gen_np_args(3000, )

target = "cuda"

mod, params =relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
            "graph", mod, tvm.cuda(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor(tvm.nd.array(priors.astype(dtype)), \
        tvm.nd.array(tblr.astype(dtype))).numpy()
