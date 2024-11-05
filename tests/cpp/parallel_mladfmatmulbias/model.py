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
import argparse

np.random.seed(42)

def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_parallel_mladfmatmul_model(M, K, N, InT, WtT, OutT):
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y0 = make_tensor_value_info("Y0", OutT, [1, M, N])
    Y1 = make_tensor_value_info("Y1", OutT, [1, M, N])

    np_wts = np.random.randint(low=-5, high=5, size=(K, N)).astype(np.uint8)
    wts_tsor_0 = make_tensor(f"weights0", onnx.TensorProto.UINT8, np_wts.shape, np_wts)
    wts_tsor_1 = make_tensor(f"weights1", onnx.TensorProto.UINT8, np_wts.shape, np_wts)

    np_bias = np.random.randint(low=-5, high=5, size=(1, N)).astype(np.float32)
    bias_tsor_0 = make_tensor(f"bias0", onnx.TensorProto.FLOAT, np_bias.shape, np_bias)
    bias_tsor_1 = make_tensor(f"bias1", onnx.TensorProto.FLOAT, np_bias.shape, np_bias)

    np_scales = np.random.randint(low=-5, high=5, size=(1, int(N*K/128))).astype(np.float32)
    scales_tsor_0 = make_tensor(f"scales0", onnx.TensorProto.FLOAT, np_scales.shape, np_scales)
    scales_tsor_1 = make_tensor(f"scales1", onnx.TensorProto.FLOAT, np_scales.shape, np_scales)

    np_zeros = np.random.randint(low=-5, high=5, size=(1, int(N*K/128))).astype(np.uint8)
    zeros_tsor_0 = make_tensor(f"zeros0", onnx.TensorProto.UINT8, np_zeros.shape, np_zeros)
    zeros_tsor_1 = make_tensor(f"zeros1", onnx.TensorProto.UINT8, np_zeros.shape, np_zeros)

    mmul0 = make_node(
        name="mmul0",
        op_type="MladfMatMul",
        inputs=["X", "weights0", "bias0", "scales0", "zeros0", ],
        outputs=["Y0"],
    )
    attr0 = onnx.helper.make_attribute("group_size", 128)
    mmul0.attribute.append(attr0)
    mmul1 = make_node(
        name="mmul1",
        op_type="MladfMatMul",
        inputs=["X", "weights1", "bias1", "scales1", "zeros1", ],
        outputs=["Y1"],
    )
    attr1 = onnx.helper.make_attribute("group_size", 128)
    mmul1.attribute.append(attr1)

    graph = make_graph(
        [mmul0, mmul1],
        "parallel_mmult",
        [X],
        [Y0, Y1],
        initializer=[wts_tsor_0, bias_tsor_0, scales_tsor_0, zeros_tsor_0, wts_tsor_1, bias_tsor_1, scales_tsor_1, zeros_tsor_1],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model, [np_wts]


if __name__ == "__main__":

    dir_name = "test_parallel_mladfmatmul"
    model_name = dir_name + "/model_parallel_mladfmatmul.onnx"
    json_name = dir_name + "/model_parallel_mladfmatmul_meta.json"
    M, K, N = (1, 4096, 11008)

    onnx_model, wts = create_parallel_mladfmatmul_model(
    M, K, N, onnx.TensorProto.BFLOAT16, onnx.TensorProto.UINT8, onnx.TensorProto.BFLOAT16)

    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{json_name}", *metainfo
    )
    print("JSON Metadata saved to", f"{json_name}")
