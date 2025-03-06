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


def create_sdslice_model(in_shape, out_shape, InT, OutT):
    X = make_tensor_value_info("X", InT, in_shape)
    Z = make_tensor_value_info("Z", OutT, out_shape)

    SDSlice_node = make_node(
        name="sd_slice",
        op_type="SDSlice",
        inputs=["X", "wts"],
        outputs=["Z"],
        domain="com.amd",
    )

    wts = [0] * 128

    wts_t = make_tensor(f"wts", onnx.TensorProto.UINT8, (128,), wts)

    graph = make_graph(
        [SDSlice_node],
        "sd_slice_graph",
        [X],
        [Z],
        initializer=[wts_t],
    )
    graph.node[0].attribute.append(onnx.helper.make_attribute("input_shape", in_shape))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("output_shape", out_shape))
    graph.node[0].attribute.append(onnx.helper.make_attribute("weight_shape", [128]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_name = "test_sdslice"
    model_name = dir_name + "/sdslice.onnx"
    json_name = dir_name + "/model_sdslice_meta.json"
    in_shape = (2, 4250, 1536)
    out_shape = (2, 4096, 1536)
    onnx_model = create_sdslice_model(
        in_shape,
        out_shape,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.BFLOAT16
    )
    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
