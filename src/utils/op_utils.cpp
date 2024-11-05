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

#include <vector>

#include "utils/op_utils.hpp"

namespace dynamic_dispatch {

namespace op_utils {

static auto OpArgMapLT = [](const OpArgMap &lhs, const OpArgMap &rhs) {
  return lhs.xrt_arg_idx < rhs.xrt_arg_idx;
};

// Input argmap contains multiple args with different xrt_arg_ids.
// Partition it to multiple slots based on each xrt_arg_id
// And sort each partition for binary search.
std::vector<std::vector<OpArgMap>>
partition_argmap(const std::vector<OpArgMap> &arg_map) {
  std::vector<std::vector<OpArgMap>> res;
  if (arg_map.size() == 0) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "Operator with arg_map size 0, skipping partition_argmap"));
    return res;
  }
  auto max_xrt_arg_id =
      *std::max_element(arg_map.begin(), arg_map.end(), OpArgMapLT);
  for (size_t i = 0; i <= max_xrt_arg_id.xrt_arg_idx; ++i) {
    std::vector<OpArgMap> args;
    std::copy_if(arg_map.begin(), arg_map.end(), std::back_inserter(args),
                 [i](const OpArgMap &arg) { return arg.xrt_arg_idx == i; });
    std::sort(args.begin(), args.end(), OpArgMapLT);
    res.push_back(std::move(args));
  }
  return res;
}

// Given an offset and xrt_arg_id, find the block(OpArg) in partition to which
// the offset belongs to. Returns reference to the corresponding OpArg
const OpArgMap &find_op_arg(const std::vector<std::vector<OpArgMap>> &argmaps,
                            size_t xrt_arg_id, size_t offset) {
  const auto &partition = argmaps.at(xrt_arg_id);
  auto iter = std::lower_bound(
      partition.begin(), partition.end(), offset,
      [](const OpArgMap &lhs, size_t val) { return lhs.offset <= val; });

  size_t idx = std::distance(partition.begin(), iter);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "find_op_arg: xrt_arg_id {} offset {} idx {}", xrt_arg_id, offset, idx));
  return argmaps.at(xrt_arg_id).at(idx - 1);
}
} // namespace op_utils
} // namespace dynamic_dispatch
