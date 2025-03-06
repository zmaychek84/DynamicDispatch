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


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_mladfsoftmax_model(x_shape, InT, WtT, OutT):
    X = make_tensor_value_info("X", InT, x_shape)
    Z = make_tensor_value_info("Z", OutT, x_shape)
    mladfsoftmax = make_node(
        name="mladfsoftmax1",
        op_type="Mladfsoftmax",
        inputs=["X", "ifm_scale", "ifm_zp", "ofm_scale", "ofm_zp", "rtp"],
        outputs=["Z"],
        domain="amd.com",
    )
    ifm_zp = [32696]
    ifm_zp_t = make_tensor(f"ifm_zp", onnx.TensorProto.UINT16, (1,), ifm_zp)
    ifm_scale = [0.00103351485449820756912231445312500000000]
    ifm_scale_t = make_tensor(f"ifm_scale", onnx.TensorProto.FLOAT, (1,), ifm_scale)

    ofm_zp = [0]
    ofm_zp_t = make_tensor(f"ofm_zp", onnx.TensorProto.UINT16, (1,), ofm_zp)
    ofm_scale = [0.00001525902189669642175853803145457732171]
    ofm_scale_t = make_tensor(f"ofm_scale", onnx.TensorProto.FLOAT, (1,), ofm_scale)

    rtp = [0] * 64
    rtp[-2] = 131
    rtp[-1] = 199
    rtp_t = make_tensor(f"rtp", onnx.TensorProto.UINT8, (64,), rtp)

    graph = make_graph(
        [mladfsoftmax],
        "SOFTMAX",
        [X],
        [Z],
        initializer=[ifm_scale_t, ifm_zp_t, ofm_scale_t, ofm_zp_t, rtp_t],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16)", required=True)

    args = parser.parse_args()
    dtype = args.dtype

    dir_name = "test_mladfsoftmax"
    model_name = dir_name + "/mladfsoftmax.onnx"
    json_name = dir_name + "/model_mladfsoftmax_meta.json"
    if dtype == "a16":
        onnx_model = create_mladfsoftmax_model(
            [4096, 4096],
            onnx.TensorProto.UINT16,
            onnx.TensorProto.UINT8,
            onnx.TensorProto.UINT16,
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(dir_name, exist_ok=True)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))

    onnx.save(onnx_model, f"{model_name}")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
