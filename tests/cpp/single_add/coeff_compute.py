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

import numpy as np


class EltwiseAdd:
    def __init__(self, a_scale, a_zp, b_scale, b_zp):
        self.a_scale = a_scale
        self.a_zp = a_zp
        self.b_scale = b_scale
        self.b_zp = b_zp
        assert isinstance(self.a_scale, float), "a_scale must be float value"
        assert isinstance(self.a_zp, int), "a_zp must be int value"

        assert isinstance(self.b_scale, float), "b_scale must be float value"
        assert isinstance(self.b_zp, int), "b_zp must be int value"

    def f2bf(self, data, bits=16):
        xf = (
            data.astype(np.float32).getfield(np.int32) & ~(2 ** (32 - bits) - 1)
        ).getfield(np.float32)
        x32 = xf.view(np.uint32)
        x16 = (x32 >> 16).astype(np.uint16)
        return x16

    def cal_coeff(self):
        co_eff1 = self.f2bf(np.asarray(self.a_scale))
        co_eff2 = self.a_zp

        co_eff3 = self.f2bf(np.asarray(self.b_scale))
        co_eff4 = self.b_zp

        return (co_eff1, co_eff2, co_eff3, co_eff4)
