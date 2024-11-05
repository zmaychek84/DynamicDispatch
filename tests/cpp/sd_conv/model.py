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
from ryzenai_dynamic_dispatch import fuse, bfp16
import argparse


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)

def read_hex_file(file_path):
    try:
        buffers = []
        with open(file_path, 'r') as file:
            for line in file:
                hex_values = line.split()
                for hex_value in hex_values:
                    buffers.append(int(hex_value, 16).to_bytes(4, byteorder='little', signed=False))
    except FileNotFoundError:
        print(f"Failed to open file {file_path}!")
        return b''
    buffer = b''.join(buffers)
    return buffer

def create_sdconv_model(ifm_info, w_info, b_info, out_info):
    X = make_tensor_value_info("X", ifm_info[0], ifm_info[1])
    Z = make_tensor_value_info("Z", out_info[0], out_info[1])
    SDConv_node = make_node(
        name="sd_conv",
        op_type="SDConv",
        inputs=["X", "w"],
        outputs=["Z"],
        domain="com.amd",
    )

    wts_float_path = "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_conv/a16bfw16bfpacc16bf_128_256_512_8_512_8_1_1_wts32.npy"
    bias_float_path = "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_conv/a16bfw16bfpacc16bf_128_256_512_8_512_8_1_1_bias32.npy"
    in_wts = np.load(wts_float_path)
    in_bias = np.load(bias_float_path)
    in_shape = np.array([ifm_info[1][1], ifm_info[1][2], out_info[1][1], out_info[1][2]])
    wts_bias_shuffle = bfp16.const_from_fp32_to_bfp16(in_wts, in_bias, "SDConv", in_shape)
    w_tensor = make_tensor("w", w_info[0], wts_bias_shuffle.shape, wts_bias_shuffle)

    graph = make_graph(
        [SDConv_node],
        "sd_conv_graph",
        [X],
        [Z],
        initializer=[w_tensor],
    )
    graph.node[0].attribute.append(onnx.helper.make_attribute("input_shape", ifm_info[1]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfp16ebs8", "float", "bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("output_shape", out_info[1]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("weight_shape", w_info[1]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dtypes", help="List of data types", nargs='+', required=False, default=["bf16", "bfp16", "float", "bf16"])

    args = parser.parse_args()
    data_types = args.dtypes

    dir_name = "test_sdconv"
    model_name = dir_name + "/sdconv.onnx"
    json_name = dir_name + "/model_sdconv_meta.json"
    if data_types[0] == "bf16" and data_types[1] == "bfp16" and data_types[2] == "float" and data_types[3] == "bf16":
        onnx_model = create_sdconv_model(
            (onnx.TensorProto.BFLOAT16,   [1, 512, 8, 256]),  # ifm [n, ih, iw, ic]
            (onnx.TensorProto.UINT8,    [128, 1, 1, 256]),  # wts [oc, kh, kw, ic]
            (onnx.TensorProto.FLOAT,    [128]),             # bias
            (onnx.TensorProto.BFLOAT16,   [1, 512, 8, 128]),  # ofm[n, oh, ow, oc]
        )
    else:
        raise ValueError(f"Unsupported dtypes: {data_types}")

    os.makedirs(dir_name, exist_ok=True)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))

    onnx.save(onnx_model, f"{model_name}")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
