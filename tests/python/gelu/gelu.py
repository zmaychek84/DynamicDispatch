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

import sys
import numpy as np
from scipy import special

from ryzenai_dynamic_dispatch import load_meta_json
from ryzenai_dynamic_dispatch import FusionRuntime

def s(x)    :
    y = 1.702*x
    return x/(1 + np.exp(-y))

def gelu(x):
    return x*(1+special.erf(x/math.sqrt(2)))/2

def gelu2(x):
    return x*(1+special.erf(x/math.sqrt(2)))/2

def q(x):
    p1 = 1.702
    q0 = 1/2
    q1 = 0.25
    q2 = 0.03125


    mul1=p1*x

    ## sigmoid
    mul1 = np.minimum(mul1,4)
    mul1 = np.maximum(mul1,-4)
    mul1_abs = np.fabs(mul1)


    mul0 = q2*mul1_abs
    mac0 = q0+mul1*q1
    msc = mac0 - mul0*mul1
    mul2 = msc*x
    return mul2

def float_to_bfloat(x):
    # View the array as int16
    int16_view = x.view(np.int16)

    # Discard every other int16 element to truncate to bfloat16
    # Since each float32 is composed of two int16 elements, we need to take every other int16 element
    return np.ascontiguousarray(int16_view[..., 1::2])

def bfloat_to_float(x):
    float32_reconstructed_int = np.zeros(x.shape, dtype=np.int32)
    float32_reconstructed_int[:] = x.astype(np.int32) << 16
    return float32_reconstructed_int.view(np.float32)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python gelu.py <path_to_meta_json>")
        sys.exit(1)

    np.random.seed(42)

    # Generate a random float32 array
    float32_input = np.random.rand(1, 1, 11008).astype(np.float32)

    bfloat16_input = float_to_bfloat(float32_input)
    golden_input = bfloat_to_float(bfloat16_input)

    inputs = [bfloat16_input]
    outputs = [np.random.randint(low=-42, high=42, size=(1, 1, 11008)).astype(np.int16)]

    #"test_silu_abf16/model_silu_meta.json"
    meta_json = sys.argv[1]
    xclbin = "xclbin/stx/gelue_4x4_abfloat16cbfloat.xclbin"

    meta_data = load_meta_json(meta_json)
    rt = FusionRuntime(xclbin)
    rt.init(meta_data)
    rt.execute(inputs, outputs)

    device_output = bfloat_to_float(outputs[0])
    golden_output = q(golden_input)

    print("golden output:", golden_output)
    print("device output:", device_output)

    # Desired relative tolerance and absolute tolerance
    rtol = 0.1    # 10%
    atol = 0.08   # Optional, but can be used for very small numbers

    # Check if all elements are close within the specified tolerance
    within_threshold = np.allclose(device_output, golden_output, rtol=rtol, atol=atol)

    print("All elements within threshold:", within_threshold)

    if not within_threshold:
        sys.exit(1)

    sys.exit(0)
