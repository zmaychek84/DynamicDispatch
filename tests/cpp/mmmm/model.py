# Copyright (c) 2025 Advanced Micro Devices, Inc
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
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse

def make_matmul_params(M, K, N, G, layer_name):
    np_wts = np.random.randint(low=-5, high=5, size=(K, N)).astype(np.uint8)
    wts_tsor = make_tensor(f"{layer_name}_weights", onnx.TensorProto.UINT8, np_wts.shape, np_wts)

    np_bias = np.random.randint(low=-5, high=5, size=(1, N)).astype(np.float32)
    bias_tsor = make_tensor(f"{layer_name}_bias", onnx.TensorProto.FLOAT, np_bias.shape, np_bias)

    np_scales = np.random.randint(low=-5, high=5, size=(1, int(N*K/G))).astype(np.float32)
    scales_tsor = make_tensor(f"{layer_name}_scales", onnx.TensorProto.FLOAT, np_scales.shape, np_scales)

    np_zeros = np.random.randint(low=-5, high=5, size=(1, int(N*K/G))).astype(np.uint8)
    zeros_tsor = make_tensor(f"{layer_name}_zeros", onnx.TensorProto.UINT8, np_zeros.shape, np_zeros)

    return wts_tsor, bias_tsor, scales_tsor, zeros_tsor

# Input of MLP is M x 4096 == X
# Output of MLP is M x 4096 == C
def create_mlp_model(M, K, N, InT):

    X = make_tensor_value_info("X", InT, [1, M, K])
    Y0 = make_tensor_value_info("Y0", InT, [1, M, N])
    Z0 = make_tensor_value_info("Z0", InT, [1, M, K])

    wts_tsor_mul1, bias_tsor_mul1, scales_tsor_mul1, zeros_tsor_mul1 = make_matmul_params(M, K, N, 128, "mul1")

    mul1 = make_node(
        name="gate_proj",
        op_type="MladfMatMul",
        inputs=["X", "mul1_weights", "mul1_bias", "mul1_scales", "mul1_zeros", ],
        outputs=["Y0"],
        domain="amd.com",
    )
    attr1 = onnx.helper.make_attribute("group_size", 128)
    mul1.attribute.append(attr1)

    # Note K and N are swapped here
    wts_tsor_mul2, bias_tsor_mul2, scales_tsor_mul2, zeros_tsor_mul2 = make_matmul_params(M, N, K, 128, "mul2")

    mul2 = make_node(
        name="down_proj",
        op_type="MladfMatMul",
        inputs=["Y0", "mul2_weights", "mul2_bias", "mul2_scales", "mul2_zeros", ],
        outputs=["Z0"],
        domain="amd.com",
    )
    attr2 = onnx.helper.make_attribute("group_size", 128)
    mul2.attribute.append(attr2)

    graph = make_graph(
        [mul1, mul2], "mmmm", [X], [Z0], value_info=[Y0]
    )
    graph.initializer.extend([wts_tsor_mul1, bias_tsor_mul1, scales_tsor_mul1, zeros_tsor_mul1])
    graph.initializer.extend([wts_tsor_mul2, bias_tsor_mul2, scales_tsor_mul2, zeros_tsor_mul2])

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19), onnx.helper.make_opsetid('amd.com', 1)])

    return onnx_model


if __name__ == "__main__":
    M, K, N = (1, 4096, 11008)
    dir_name = "test_mmmm_abf16"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model = create_mlp_model(M, K, N, onnx.TensorProto.BFLOAT16)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, f"{dir_name}/model_mmmm.onnx")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_mmmm_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_mmmm_meta.json")
