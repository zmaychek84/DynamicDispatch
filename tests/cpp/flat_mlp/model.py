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
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse

def make_matmul_params(M, K, N, G, layer_name):
    np_wts = np.random.randint(low=0, high=255, size=(N, int(K/G), int(G/2))).astype(np.uint8)
    wts_tsor = make_tensor(f"{layer_name}_wts", onnx.TensorProto.UINT8, np_wts.shape, np_wts)

    np_bias = np.random.uniform(0, 1, size=(int(N))).astype(np.float32)
    bias_tsor = make_tensor(f"{layer_name}_bias", onnx.TensorProto.FLOAT, np_bias.shape, np_bias)

    np_scales = np.random.uniform(-1, 1, size=(int(N* int(K/G)))).astype(np.float32)
    scales_tsor = make_tensor(f"{layer_name}_scale", onnx.TensorProto.FLOAT, np_scales.shape, np_scales)

    np_zeros = np.random.randint(low=0, high=255, size=(int(N* int(K/G/2)))).astype(np.uint8)
    zeros_tsor = make_tensor(f"{layer_name}_zp", onnx.TensorProto.UINT8, np_zeros.shape, np_zeros)

    return wts_tsor, bias_tsor, scales_tsor, zeros_tsor

def create_mlp_model(M, K, N, G):
    ifm = make_tensor_value_info("ifm", onnx.TensorProto.BFLOAT16, [1, M, K])
    ofm = make_tensor_value_info("ofm", onnx.TensorProto.BFLOAT16, [1, M, N])

    gate_wts_tensor, gate_bias_tensor, gate_scale_tensor, gate_zp_tensor = make_matmul_params(M, K, N, 128, "gate")
    up_wts_tensor, up_bias_tensor, up_scale_tensor, up_zp_tensor = make_matmul_params(M, K, N, 128, "up")

    fused_mlp = make_node(
        name="flat_mlp",
        op_type="FlatMLP",
        inputs=["ifm", "gate_wts", "gate_scale", "gate_zp", "gate_bias",
                "up_wts", "up_scale", "up_zp", "up_bias"],
        outputs=["ofm"],
        domain="com.amd",
    )

    graph = make_graph(
        [fused_mlp],
        "fused_mlp_graph",
        [ifm],
        [ofm],
        initializer = [gate_wts_tensor, gate_bias_tensor, gate_scale_tensor, gate_zp_tensor,
                       up_wts_tensor, up_bias_tensor, up_scale_tensor, up_zp_tensor]
    )
    graph.node[0].attribute.append(onnx.helper.make_attribute("input_shape", [M, K, N]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("group_size", 128))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16" ,"uint8" ,"float" ,"uint8" ,"float" ,"uint8" ,"float" ,"uint8" ,"float"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model


if __name__ == "__main__":
    M, K, N = (1, 3072, 8192)
    G = 128
    dir_name = "test_flat_mlp"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model = create_mlp_model(M, K, N, G)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{dir_name}/flat_mlp.onnx")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/flat_mlp_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_mlp_meta.json")
