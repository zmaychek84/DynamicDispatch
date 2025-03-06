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
    make_attribute
)
from onnx.checker import check_model
import onnxruntime
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


# Z = X*Y
def create_quantizelinear(B, M, N, LhsT, OutT, sc, zp):
    X = make_tensor_value_info("X", LhsT, [B, M, N])
    Z = make_tensor_value_info("Z", OutT, [B, M, N])

    sc_tsr = make_tensor("SC", onnx.TensorProto.FLOAT, sc.shape, sc)
    zp_tsr = make_tensor("ZP", onnx.TensorProto.UINT16, zp.shape, zp)

    quantizelinear = make_node(
        name="quantizelinear",
        op_type="QLinear_CPU",
        inputs=["X", "SC", "ZP"],
        outputs=["Z"],
        domain="amd.com",
    )
    new_attr = make_attribute("input_shape", [B, M, N])
    quantizelinear.attribute.append(new_attr)

    graph = make_graph(
        [quantizelinear], "quantizelinear", [X], [Z], initializer=[sc_tsr,zp_tsr]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    B, M, N = (12, 64, 512)
    dir_name = "test_quantizelinear_cpu"
    os.makedirs(dir_name, exist_ok=True)

    scale_ = np.array([0.0487],dtype=float)
    zp_ =  np.array([24],dtype=int)
    onnx_model = create_quantizelinear(
        B, M, N, onnx.TensorProto.FLOAT, onnx.TensorProto.UINT16, scale_, zp_
    )
    onnx.save(onnx_model, f"{dir_name}/model_quantizelinear_cpu.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_quantizelinear_cpu_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_quantizelinear_cpu_meta.json")
