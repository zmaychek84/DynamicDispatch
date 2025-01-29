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
    make_attribute,
)
from onnx.checker import check_model
import onnxruntime
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse
import argparse

def create_single_mha_model(B, M, K, N, heads, InT):
    Q_info = make_tensor_value_info("Q", InT, [B, M, K])
    K_info = make_tensor_value_info("K", InT, [B, N, K])
    V_info = make_tensor_value_info("V", InT, [B, N, K])
    Y_info = make_tensor_value_info("Y", InT, [B, M, K])

    # mask need to >= 128 elements
    padded_N = N if N >= 128 else 128
    mask_np = np.zeros((padded_N), dtype=np.uint16)
    mask_tsor = make_tensor("mask", onnx.TensorProto.UINT16, mask_np.shape, mask_np)

    sd_mha_node = make_node(
        name="sd_mha",
        op_type="SDMHA",
        inputs=["Q", "K", "V", "mask"],
        outputs=["Y"],
        domain="com.amd",
    )
    sd_mha_node.attribute.append(onnx.helper.make_attribute("input_shape", [B, M, K, N]))
    sd_mha_node.attribute.append(onnx.helper.make_attribute("num_heads", heads)) # list of int or int are both converted to vector[int]
    sd_mha_node.attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfloat16", "bfloat16"]))
    sd_mha_node.attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))

    graph = make_graph(
        [sd_mha_node], "sd_mha_graph", [Q_info, K_info, V_info], [Y_info], initializer=[mask_tsor]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_name = "test_sdmha"
    model_name = dir_name + "/sdmha.onnx"
    json_name = dir_name + "/model_sdmha_meta.json"
    B, M, K, N = (2, 1024, 640, 1024)
    onnx_model = create_single_mha_model(
        B,
        M,
        K,
        N,
        8, # heads
        onnx.TensorProto.BFLOAT16,
    )
    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
