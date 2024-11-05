// Copyright (c) 2024 Advanced Micro Devices, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <array>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ops/op_interface.hpp>
#include <stdlib.h>
#include <vector>

using json = nlohmann::json;

std::vector<char> patch_ctrl_bin(std::vector<char> const &ctrl_bin,
                                 std::vector<CtrlPktPatchInfo> ctrlpkt_info,
                                 std::vector<uint64_t> &buffer_addrs,
                                 bool verbose = false) {
  std::vector<char> patch_ctrl(ctrl_bin);

  for (auto &patch : ctrlpkt_info) {

    if (patch.xrt_arg_idx < 0 || patch.xrt_arg_idx > 3) {
      throw std::runtime_error("Unknow xrt_arg_idx");
    }
    uint64_t ddr_addr = patch.bo_offset + buffer_addrs[patch.xrt_arg_idx];
    memcpy(&patch_ctrl[patch.offset], &ddr_addr, patch.size);
  }

  return patch_ctrl;
}
