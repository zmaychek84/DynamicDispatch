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

# from cal_coeff import MatMul
np.random.seed(42)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_single_matmul_model(M, K, N, c0, InT, WtT, OutT):

    X = make_tensor_value_info("X", InT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, N])

    # wts = np.load("tensor.npy")
    # wts = np.random.randint(low=0, high=32, size=(K, N)).astype(np.uint8)
    wts = np.eye(K, dtype=np.uint8)

    # np_wts[...] = 1
    wts_tsor = make_tensor(f"W0", WtT, wts.shape, wts)

    # bias = np.random.uniform(-1, 1, wts.shape[1])

    # c0 = np.random.randint(0, 32, size=(1 * N)).astype(np.int64)
    qdq_tsor = make_tensor(f"qdq", onnx.TensorProto.INT64, c0.shape, c0)

    # np_qdq_params = np.random.randint(-16, 16, size=(16)).astype(np.int32)
    np_qdq_params = np.zeros(16).astype(np.int32)
    np_qdq_params[0] = 0
    np_qdq_params[1] = 0  # c1
    np_qdq_params[2] = 0  # c2
    np_qdq_params[3] = 0
    np_qdq_params[4] = 0
    np_qdq_params[5] = 32
    np_qdq_params[6] = 0
    np_qdq_params[7] = 0

    qdq_params_tsor = make_tensor(
        f"qdq_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )

    mul1 = make_node(
        name="mul1",
        op_type="MatMul_CPU",
        inputs=["X", "W0", "qdq", "qdq_params"],
        outputs=["Y"],
    )
    new_attr = make_attribute("input_shape", [1, M, K])
    mul1.attribute.append(new_attr)

    graph = make_graph(
        [mul1], "lr", [X], [Y], initializer=[wts_tsor, qdq_tsor, qdq_params_tsor]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    # check_model(onnx_model)
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # check_model(shape_inferred_model)
    return shape_inferred_model, [wts, c0, np_qdq_params]


def create_double_matmul_model(M, K, N, P, c0, c1, InT, WtT, OutT):
    # Input tensors
    X = make_tensor_value_info("X", InT, [1, M, K])
    Z = make_tensor_value_info("Z", OutT, [1, M, P])

    # wts1 = np.random.randint(low=0, high=32, size=(K, N)).astype(np.uint8)
    # wts2 = np.random.randint(low=0, high=32, size=(N, P)).astype(np.uint8)
    wts1 = np.eye(K, dtype=np.uint8)
    wts2 = np.eye(K, dtype=np.uint8)

    wts_tsor1 = make_tensor(f"W0", WtT, wts1.shape, wts1)
    wts_tsor2 = make_tensor(f"W1", WtT, wts2.shape, wts2)

    Y1 = make_tensor_value_info("Y1", OutT, [1, M, N])
    Y2 = make_tensor_value_info("Y2", OutT, [1, M, P])


    qdq_tsor_0 = make_tensor("qdq", onnx.TensorProto.INT64, c0.shape, c0)
    qdq_tsor_1 = make_tensor("qdq1", onnx.TensorProto.INT64, c1.shape, c1)

    # np_qdq_params = np.random.randint(-16, 16, size=(16)).astype(np.int32)
    np_qdq_params = np.zeros(16).astype(np.int32)
    np_qdq_params[0] = 0
    np_qdq_params[1] = 0  # c1
    np_qdq_params[2] = 0  # c2
    np_qdq_params[3] = 0
    np_qdq_params[4] = 0
    np_qdq_params[5] = 32
    np_qdq_params[6] = 0
    np_qdq_params[7] = 0

    qdq_params_tsor = make_tensor(
        f"qdq_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )
    matmul1 = make_node(
        name="matmul1",
        op_type="MatMul",
        inputs=["X", "W0", "qdq", "qdq_params"],
        outputs=["Y1"],
    )
    matmul2 = make_node(
        name="matmul2",
        op_type="MatMul",
        inputs=["Y1", "W1", "qdq1", "qdq_params"],
        outputs=["Y2"],
    )

    new_attr = make_attribute("input_shape", [1, M, K])
    matmul1.attribute.append(new_attr)

    new_attr = make_attribute("input_shape", [1, M, P])
    matmul2.attribute.append(new_attr)

    # Graph definition
    graph = make_graph(
        [matmul1, matmul2],
        "double_matmul_graph",
        [X],
        [Y2],
        initializer=[wts_tsor1, wts_tsor2, qdq_tsor_0, qdq_tsor_1, qdq_params_tsor],
        value_info=[Y1]
    )
    # Create the model
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    return shape_inferred_model, [wts1, wts2]

def create_triple_matmul_model(M, K, N, P, Q, c0, c1, c2, InT, WtT, OutT):

    X = make_tensor_value_info("X", InT, [1, M, K])
    Z = make_tensor_value_info("Y3", OutT, [1, M, P])

    # wts1 = np.random.randint(low=0, high=32, size=(K, N)).astype(np.uint8)
    # wts2 = np.random.randint(low=0, high=32, size=(N, P)).astype(np.uint8)
    # wts3 = np.random.randint(low=0, high=32, size=(P, Q)).astype(np.uint8)
    wts1 = np.eye(K, dtype=np.uint8)
    wts2 = np.eye(N, dtype=np.uint8)
    wts3 = np.eye(P, dtype=np.uint8)

    wts_tsor1 = make_tensor("W0", WtT, wts1.shape, wts1)
    wts_tsor2 = make_tensor("W1", WtT, wts2.shape, wts2)
    wts_tsor3 = make_tensor("W2", WtT, wts3.shape, wts3)

    Y1 = make_tensor_value_info("Y1", OutT, [1, M, N])
    Y2 = make_tensor_value_info("Y2", OutT, [1, M, P])
    Y3 = make_tensor_value_info("Y3", OutT, [1, M, Q])

    qdq_tsor_0 = make_tensor("qdq", onnx.TensorProto.INT64, c0.shape, c0)
    qdq_tsor_1 = make_tensor("qdq1", onnx.TensorProto.INT64, c1.shape, c1)
    qdq_tsor_2 = make_tensor("qdq2", onnx.TensorProto.INT64, c2.shape, c2)

    np_qdq_params = np.zeros(16).astype(np.int32)
    np_qdq_params[0] = 0
    np_qdq_params[1] = 0  # c1
    np_qdq_params[2] = 0 # c2
    np_qdq_params[3] = 0
    np_qdq_params[4] = 0
    np_qdq_params[5] = 32
    np_qdq_params[6] = 0
    np_qdq_params[7] = 0

    qdq_params_tsor = make_tensor(
        "qdq_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )

    matmul1 = make_node(
        name="matmul1",
        op_type="MatMul",
        inputs=["X", "W0", "qdq", "qdq_params"],
        outputs=["Y1"],
    )
    matmul2 = make_node(
        name="matmul2",
        op_type="MatMul",
        inputs=["Y1", "W1", "qdq1", "qdq_params"],
        outputs=["Y2"],
    )
    matmul3 = make_node(
        name="matmul3",
        op_type="MatMul_CPU",
        inputs=["Y2", "W2", "qdq2", "qdq_params"],
        outputs=["Y3"],
    )

    new_attr = make_attribute("input_shape", [1, M, K])
    matmul1.attribute.append(new_attr)

    new_attr = make_attribute("input_shape", [1, M, N])
    matmul2.attribute.append(new_attr)

    new_attr = make_attribute("input_shape", [1, M, P])
    matmul3.attribute.append(new_attr)

    graph = make_graph(
        [matmul1, matmul2, matmul3],
        "triple_matmul_graph",
        [X],
        [Y3],
        initializer=[wts_tsor1, wts_tsor2, wts_tsor3, qdq_tsor_0, qdq_tsor_1, qdq_tsor_2, qdq_params_tsor],
        value_info=[Y1, Y2]
    )
    # Create the model
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    return shape_inferred_model, [wts1, wts2, wts3]


if __name__ == "__main__":

    single_dir = "test_matmul_cpu_single/"
    double_dir = "test_matmul_cpu_double/"
    triple_dir = "test_matmul_cpu_triple/"

    single_name =  "model_single_matmul1_cpu"
    double_name =  "model_double_matmul1_cpu"
    triple_name =  "model_triple_matmul1_cpu"

    single_json = single_dir + single_name + ".json"
    double_json = double_dir + double_name + ".json"
    triple_json = triple_dir + triple_name + ".json"

    M = 128
    K, N, P, Q = (768, 768, 768, 768)
    c0 = np.random.randint(0, 32, size=(1 * N)).astype(np.int64)
    c1 = np.random.randint(0, 32, size=(1 * P)).astype(np.int64)
    c2 = np.random.randint(0, 32, size=(1 * Q)).astype(np.int64)

    single_onnx_model, single_wts = create_single_matmul_model(
        M,
        K,
        N,
        c0,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
    )

    double_onnx_model, double_wts = create_double_matmul_model(
            M,
            K,
            N,
            P,
            c0,
            c1,
            onnx.TensorProto.UINT16,
            onnx.TensorProto.UINT8,
            onnx.TensorProto.UINT16,
        )

    triple_onnx_model, triple_wts = create_triple_matmul_model(
        M,
        K,
        N,
        P,
        Q,
        c0,
        c1,
        c2,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
    )




    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(double_dir, exist_ok=True)
    os.makedirs(triple_dir, exist_ok=True)

    onnx.save(single_onnx_model, f"{single_dir}{single_name}.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(single_onnx_model), single_dir)
    json_str = fuse.save_tensors_to_json(f"{single_json}", *metainfo)
    print("JSON Metadata saved to", f"{single_json}")


    onnx.save(double_onnx_model, f"{double_dir}{double_name}.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(double_onnx_model), double_dir)
    json_str = fuse.save_tensors_to_json(f"{double_json}", *metainfo)
    print("JSON Metadata saved to", f"{double_json}")


    onnx.save(triple_onnx_model, f"{triple_dir}{triple_name}.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(triple_onnx_model), triple_dir)
    json_str = fuse.save_tensors_to_json(f"{triple_json}", *metainfo)
    print("JSON Metadata saved to", f"{triple_json}")
