# Copyright (c) OpenComputeLab. All Rights Reserved.
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args

onnx_model = onnx.load("./carl_loss.onnx")


cls_score, labels, bbox_pred, bbox_targets = gen_np_args(32, 4)

target = "cuda"

mod, params =relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=1):
    executer = relay.build_module.create_executor(
            "graph", mod, tvm.cuda(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor(tvm.nd.array(cls_score.astype(dtype)), \
        tvm.nd.array(labels.astype(dtype)), tvm.nd.array(bbox_pred.astype("int64")), \
        tvm.nd.array(bbox_targets.astype(dtype))).numpy()
