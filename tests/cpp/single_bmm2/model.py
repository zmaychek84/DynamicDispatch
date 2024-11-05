# Copyright (c) 2024 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import numpy as np
import onnx
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model
import onnxruntime
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)

def create_bmm2_model(M, K, N, B, InT, InTB , OutT, trans):
    X = make_tensor_value_info("X", InT, [B , M, K])
    if(trans == True):
        Y = make_tensor_value_info("Y", InTB, [B , K, N])
    else:
        Y = make_tensor_value_info("Y", InTB, [B , K, N])  # <<<< or is it B N K ?
    Z = make_tensor_value_info("Z", OutT, [B , M, N])

    bmm = make_node(
        name="bmm",
        op_type="BMM2",
        inputs=["X", "Y"],
        outputs=["Z"],
        domain="amd.com",
    )

    graph = make_graph(
        [bmm], "lr", [X, Y], [Z], initializer=[]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model

if __name__ == "__main__":

    M, K, N, B = (2048, 2048, 128, 32)
    dir_name = "test_bmm2_abf16"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model = create_bmm2_model(
        M, K, N, B, onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16, False
    )
    onnx.save(onnx_model, f"{dir_name}/model_bmm2.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_bmm2_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_bmm2_meta.json")
