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
from onnx import mapping
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse
import argparse
import struct
from functools import reduce

np.random.seed(0)

def create_mul_model(input_shapes, InT, WtT, OutT):
    K = input_shapes[0]
    M = input_shapes[1]


    X = make_tensor_value_info("X", InT, [K, M])
    Y = make_tensor_value_info("Y", OutT, [K, M])

    qdq =  np.random.randint(low=0, high=255, size=[16]).astype(int)
    qdq_tsor = make_tensor(f"QDQ", onnx.TensorProto.INT32, [16], qdq)

    mul1 = make_node(
        name="LpNorm",
        op_type="L2_Norm",
        inputs=["X", "QDQ"],
        outputs=["Y"],
        domain="com.amd",
    )
    graph = make_graph([mul1], "L2_Norm", [X], [Y], initializer=[qdq_tsor])

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_l2_norm"
    model_name = dir_name + "/l2_norm.onnx"
    json_name = dir_name + "/model_l2_norm_meta.json"

    if dtype == 'a16':
        onnx_model = create_mul_model(
        [64, 3072], onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
