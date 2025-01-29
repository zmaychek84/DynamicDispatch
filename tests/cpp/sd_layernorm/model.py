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

def aie_srs(input, aie_mode):
    '''
    AIE_SRS Applies shift-round-saturate operation as in ME to input data
      Applies shift-round-saturate operation as per ME 16-bit finite
      precision arithmetic. Input is assumed to be integer-valued. The
      processing consists of three parts
          - Shifting down of the data by the number of bits specified in
          shift
          - Rounding according to mode
              0 - truncation
              2 - to nearest with 'half' rounded up (towars +infinity)
              6 - to nearest with 'half' rounded to even
          - Saturation of values exceeding the data_width dynamic range
    Arguments:
    input: Input data/array to AIE SRS
    aie_mode: AIE SRS mode parameters [data_width, shift, mode, dtype]
    Return:
    output: Shifted output data
    '''

    #Assign aiemode parameters into variables
    data_width = aie_mode[0]
    shift = aie_mode[1]
    mode = aie_mode[2]
    unsigned = aie_mode[3]

    #Shift
    output = input/(2**(shift))
    #Round
    if (mode == 0):
        # simply truncate LSBs
        output = np.fix(output) #mimics the functionality of aie::floor - https://jira.xilinx.com/browse/CR-1166356
    elif (mode == 2):
        # round to nearest, round half up (towards +infinity)
        output = np.round(output)
        # detect negative values at the exact half point (-0.5) and add -0.5 before rounding
        i = ((input - np.fix(input)) == -0.5)
        output[i] = np.round(input + 0.5)[i]
    elif (mode == 6):
        # round to nearest, round half to even
        output = np.around(output)
    else:
        print('Unexpected rounding mode')

    if(unsigned == 0):
        # positive clip
        output[(output >= 2**(data_width-1) - 1)] = 2**(data_width-1) - 1
        # negative clip
        output[(output <= -1*2**(data_width-1))] = -1*2**(data_width-1)
    else:
        output[(output>2**(data_width) - 1)] = 2**(data_width) - 1
        output[(output<0)] = 0

    return output

def np_fp32_2_bf16(x:np.ndarray):
    x.dtype = np.uint32
    x = aie_srs(x, [16, 16, 6, 1])
    x = x.astype(np.uint16)
    # x.dtype = np.float32
    return x

def create_single_matmul_model(B, M, K,  InT, WtT, OutT):
    X = make_tensor_value_info("X", InT, [B, M, K])
    Y = make_tensor_value_info("Y", OutT, [B, M, K])

    gamma = np_fp32_2_bf16(np.random.uniform(low=-1.0, high=1.0, size=(K)).astype(np.float32))
    gamma_tensor = make_tensor("gamma", WtT, gamma.shape, gamma)

    b = np_fp32_2_bf16(np.random.uniform(low=-1.0, high=1.0, size=(K)).astype(np.float32))
    b_tensor = make_tensor("b", WtT, b.shape, b)

    sd_matmul_node = make_node(
        name="sd_layernorm",
        op_type="SDLayerNorm",
        inputs=["X", "gamma", "b"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph(
        [sd_matmul_node], "sd_layernorm_graph", [X], [Y], initializer=[gamma_tensor, b_tensor],
    )
    graph.node[0].attribute.append(onnx.helper.make_attribute("input_shape", [B, M, K]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfloat16", "bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("output_shape", [B, M, K]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("gamma_shape", gamma.shape))
    graph.node[0].attribute.append(onnx.helper.make_attribute("beta_shape", b.shape))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_name = "test_sdlayernorm"
    model_name = dir_name + "/sdlayernorm.onnx"
    json_name = dir_name + "/model_sdlayernorm_meta.json"
    B, M, K = (2, 4096, 320)
    onnx_model = create_single_matmul_model(
        B,
        M,
        K,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.BFLOAT16
    )
    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
