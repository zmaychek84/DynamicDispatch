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

def create_single_mha_model(num_heads, seq_len_q, seq_len_total_k, head_size, InT):
    #input q k v
    ifm = make_tensor_value_info("ifm", InT, [seq_len_q, num_heads, head_size])
    dummy_ifm1 = make_tensor_value_info("dummy_ifm1", InT, [seq_len_q, num_heads, head_size])
    dummy_ifm2 = make_tensor_value_info("dummy_ifm2", InT, [seq_len_q, num_heads, head_size])
    passed_k = make_tensor_value_info("passed_k", InT, [1, num_heads, seq_len_total_k, head_size])
    passed_v = make_tensor_value_info("passed_v", InT, [1, num_heads, seq_len_total_k, head_size])
    passed_seq_len = make_tensor_value_info("passed_seq_len", onnx.TensorProto.INT32, [1, 1])
    cur_seq_len = make_tensor_value_info("cur_seq_len", onnx.TensorProto.INT32, [1])
    rope_cos_wts_value = np.random.uniform(low=-1.0, high=1.0, size=(seq_len_total_k, int(head_size/2))).astype(np.float32)
    rope_sin_wts_value = np.random.uniform(low=-1.0, high=1.0, size=(seq_len_total_k, int(head_size/2))).astype(np.float32)
    cos_cache = make_tensor("cos_cache", onnx.TensorProto.FLOAT, rope_cos_wts_value.shape, rope_cos_wts_value)
    sin_cache = make_tensor("sin_cache", onnx.TensorProto.FLOAT, rope_sin_wts_value.shape, rope_sin_wts_value)

    ofm = make_tensor_value_info("ofm", InT, [seq_len_q, num_heads, head_size])
    present_k = make_tensor_value_info("present_k", InT, [1, num_heads, seq_len_total_k, head_size])
    present_v = make_tensor_value_info("present_v", InT, [1, num_heads, seq_len_total_k, head_size])

    flat_mha_node = make_node(
        name="flat_mha",
        op_type="FLATMHA",
        inputs=["ifm", "dummy_ifm1", "dummy_ifm2", "passed_k", "passed_v", "passed_seq_len", "cur_seq_len", "cos_cache", "sin_cache"],
        outputs=["ofm", "present_k", "present_v"],
        domain="com.amd",
    )
    flat_mha_node.attribute.append(onnx.helper.make_attribute("input_shape", [num_heads, seq_len_q, seq_len_total_k, head_size]))
    flat_mha_node.attribute.append(onnx.helper.make_attribute("kv_num_heads", num_heads))
    flat_mha_node.attribute.append(onnx.helper.make_attribute("num_heads", num_heads))
    flat_mha_node.attribute.append(onnx.helper.make_attribute("rotary_interleaved", 0))
    flat_mha_node.attribute.append(onnx.helper.make_attribute("scale", 0.10206207633018494))

    graph = make_graph(
        [flat_mha_node], "flat_mha_graph", [ifm, dummy_ifm1, dummy_ifm2, passed_k, passed_v, passed_seq_len, cur_seq_len], [ofm, present_k, present_v], initializer=[cos_cache, sin_cache]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_name = "test_flat_mha_v2"
    model_name = dir_name + "/flat_mha.onnx"
    json_name = dir_name + "/model_flat_mha_meta.json"
    num_heads, seq_len_q, seq_len_total_k, head_size = (32, 1, 1024, 96)
    onnx_model = create_single_mha_model(num_heads, seq_len_q, seq_len_total_k, head_size, onnx.TensorProto.BFLOAT16)
    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
