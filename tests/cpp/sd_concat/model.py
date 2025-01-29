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

def create_concat_model(ifm1_info, ifm2_info, out_info):
    a_shape = ifm1_info[1]
    b_shape = ifm2_info[1]
    c_shape = out_info[1]
    A = make_tensor_value_info("A", ifm1_info[0], ifm1_info[1])
    B = make_tensor_value_info("B", ifm2_info[0], ifm2_info[1])
    C = make_tensor_value_info("C", out_info[0], out_info[1])
    SDconcat_node = make_node(
        name="sd_concat",
        op_type="SDConcat",
        inputs=["A", "B"],
        outputs=["C"],
        domain="com.amd",
    )

    graph = make_graph(
        [SDconcat_node],
        "sd_concat_graph",
        [A, B],
        [C],
    )
    graph.node[0].attribute.append(onnx.helper.make_attribute("a_shape", ifm1_info[1]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("b_shape", ifm2_info[1]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("c_shape", out_info[1]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dtypes", help="List of data types", nargs='+', required=False, default=["bf16", "bfp16", "bf16"])

    args = parser.parse_args()
    data_types = args.dtypes
    a_shape = [2, 8, 8, 1280]
    b_shape = [2, 8, 8, 1280]
    c_shape = [2, 8, 8, 2560]

    dir_name = "test_sd_concat"
    model_name = dir_name + "/sd_concat.onnx"
    json_name = dir_name + "/model_sd_concat_meta.json"
    if data_types[0] == "bf16" and data_types[1] == "bfp16" and data_types[2] == "bf16":
        onnx_model = create_concat_model(
            (onnx.TensorProto.BFLOAT16,   a_shape),  # ifm 1
            (onnx.TensorProto.BFLOAT16,   b_shape),  # ifm 2
            (onnx.TensorProto.BFLOAT16,  c_shape),  # ofm
        )
    else:
        raise ValueError(f"Unsupported dtypes: {data_types}")

    os.makedirs(dir_name, exist_ok=True)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))

    onnx.save(onnx_model, f"{model_name}")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
