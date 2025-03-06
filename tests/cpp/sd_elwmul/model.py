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

def create_elwmul_model(ifm1_info, ifm2_info, out_info):
    a_shape = ifm1_info[1]
    b_shape = ifm2_info[1]
    c_shape = out_info[1]
    A = make_tensor_value_info("A", ifm1_info[0], ifm1_info[1])
    B = make_tensor_value_info("B", ifm2_info[0], ifm2_info[1])
    C = make_tensor_value_info("C", out_info[0], out_info[1])
    SDelwmul_node = make_node(
        name="sd_elwmul",
        op_type="SDMul",
        inputs=["A", "B"],
        outputs=["C"],
        domain="com.amd",
    )

    if len(a_shape) != len(b_shape):
        wts = np_fp32_2_bf16(np.random.uniform(low=-1.0, high=1.0, size=b_shape).astype(np.float32))
        wts_tsor = make_tensor("B", onnx.TensorProto.UINT16, wts.shape, wts)
        inputs = [A]
        wts_initializer=[wts_tsor]
    else:
        inputs = [A, B]
        wts_initializer = None

    graph = make_graph(
        [SDelwmul_node],
        "sd_elwmul_graph",
        inputs,
        [C],
        initializer=wts_initializer
    )
    graph.node[0].attribute.append(onnx.helper.make_attribute("a_shape", ifm1_info[1]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("c_shape", out_info[1]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("b_shape", ifm2_info[1]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dtypes", help="List of data types", nargs='+', required=False, default=["bf16", "bfp16", "bf16"])
    parser.add_argument("--wts", action="store_true", help="use ifm + wts form")

    args = parser.parse_args()
    data_types = args.dtypes
    # if args.wts:
    #     a_shape = [2, 4096, 320]
    #     b_shape = [320]
    # else:
    #     a_shape = [2, 8, 8, 1280]
    #     b_shape = [2, 8, 8, 1280]
    a_shape = [2, 4096, 1280]
    b_shape = [2, 4096, 1280]
    dir_name = "test_sd_elwmul"
    model_name = dir_name + "/sd_elwmul.onnx"
    json_name = dir_name + "/model_sd_elwmul_meta.json"
    if data_types[0] == "bf16" and data_types[1] == "bfp16" and data_types[2] == "bf16":
        onnx_model = create_elwmul_model(
            (onnx.TensorProto.BFLOAT16,   a_shape),  # ifm 1
            (onnx.TensorProto.BFLOAT16,   b_shape),  # ifm 2
            (onnx.TensorProto.BFLOAT16,  a_shape),  # ofm
        )
    else:
        raise ValueError(f"Unsupported dtypes: {data_types}")

    os.makedirs(dir_name, exist_ok=True)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))

    onnx.save(onnx_model, f"{model_name}")

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *metainfo)
    print("JSON Metadata saved to", f"{json_name}")
