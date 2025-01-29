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

np.random.seed(42)



def create_single_groupnorm_model(B, H, W, C, InT, WtT, OutT):
    X = make_tensor_value_info("X", InT, [B, H, W, C])
    Y = make_tensor_value_info("Y", OutT, [B, H, W, C])

    gamma = np.random.randint(0, 65536, size=C, dtype=np.uint16)
    beta = np.random.randint(0, 65536, size=C, dtype=np.uint16)
    wts = np.concatenate((gamma, beta))
    wts_tsor = make_tensor("wts", WtT, wts.shape, wts)

    sd_gn_node = make_node(
        name="sd_groupnorm",
        op_type="SDGroupNorm",
        inputs=["X", "wts"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph(
        [sd_gn_node], "sd_matmul_graph", [X], [Y], initializer=[wts_tsor],
    )

    graph.node[0].attribute.append(onnx.helper.make_attribute("input_shape", [B, H, W, C]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("output_shape", [B, H, W, C]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("wts_shape", [C * 2]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_name = "test_sdgroupnorm"
    model_name = dir_name + "/sdgroupnorm.onnx"
    json_name = dir_name + "/model_sdgroupnorm_meta.json"
    B, H, W, C = (1, 128, 128, 512)
    onnx_model = create_single_groupnorm_model(
        B,
        H,
        W,
        C,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.BFLOAT16
    )
    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
