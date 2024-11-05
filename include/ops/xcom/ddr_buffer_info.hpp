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

#include <cstdint>

namespace ryzenai {
namespace xcom {

struct ddr_buffer_info_s {
  std::int64_t ifm_addr;
  std::int64_t ifm_size;
  std::int64_t param_addr;
  std::int64_t param_size;
  std::int64_t ofm_addr;
  std::int64_t ofm_size;
  std::int64_t inter_addr;
  std::int64_t inter_size;
  std::int64_t mc_code_addr;
  std::int64_t mc_code_size;
  std::int64_t pad_control_packet;
};

static_assert(sizeof(ddr_buffer_info_s) == 11 * sizeof(std::int64_t));

} // namespace xcom
} // namespace ryzenai
