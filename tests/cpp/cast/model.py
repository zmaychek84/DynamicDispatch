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


def create_resize_model(ifm_info, out_info):
    A = make_tensor_value_info("A", ifm_info[0], ifm_info[1])
    C = make_tensor_value_info("C", out_info[0], out_info[1])
    cast_node = make_node(
        name="cast",
        op_type="Cast",
        inputs=["A"],
        outputs=["C"],
        domain="com.amd",
    )

    graph = make_graph(
        [cast_node],
        "cast_graph",
        [A],
        [C],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dtypes",
        help="List of data types",
        nargs="+",
        required=False,
        default=["bf16", "bfp16"],
    )
    parser.add_argument("--H", type=int, default=2048, help="H")
    parser.add_argument("--W", type=int, default=4096, help="W")

    args = parser.parse_args()
    data_types = args.dtypes
    h = args.H
    w = args.W

    a_shape = [h, w]
    c_shape = [int(h / 8), int(w / 8), 8, 9]

    dir_name = "test_cast"
    model_name = dir_name + "/cast.onnx"
    json_name = dir_name + "/model_cast_meta.json"
    if data_types[0] == "bf16" and data_types[1] == "bfp16":
        onnx_model = create_resize_model(
            (onnx.TensorProto.BFLOAT16, a_shape),
            (onnx.TensorProto.UINT8, c_shape),
        )
    else:
        raise ValueError(f"Unsupported dtypes: {data_types}")

    os.makedirs(dir_name, exist_ok=True)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))

    onnx.save(onnx_model, f"{model_name}")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
