
// Copyright (c) 2025 Advanced Micro Devices, Inc
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

#include <op_fuser/fuse_types.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

void sync_identity_tensors(Metadata &meta) {
  // backward iterate and update identity ops except those with input
  // parent_name
  for (auto it = meta.op_list.rbegin(); it != meta.op_list.rend(); ++it) {
    const auto &op = *it;
    if (op.type == "Identity") {
      auto args = OpsFusion::get_op_args(op);
      auto &input_tensor = MAP_AT(meta.tensor_map, args[0]);
      auto &output_tensor = MAP_AT(meta.tensor_map, args[1]);
      if (input_tensor.parent_name != "in") {
        input_tensor.arg_idx = output_tensor.arg_idx;
        input_tensor.parent_name = output_tensor.parent_name;
        input_tensor.offset = output_tensor.offset;
      }
    }
  }
  // forward iterate to consider more than one identity ops back to back at
  // input
  for (auto &op : meta.op_list) {
    if (op.type == "Identity") {
      auto args = OpsFusion::get_op_args(op);
      auto &input_tensor = MAP_AT(meta.tensor_map, args[0]);
      auto &output_tensor = MAP_AT(meta.tensor_map, args[1]);
      if (input_tensor.parent_name == "in") {
        output_tensor.arg_idx = input_tensor.arg_idx;
        output_tensor.parent_name = input_tensor.parent_name;
        output_tensor.offset = input_tensor.offset;
      }
    }
  }
  RYZENAI_LOG_TRACE("Sync Identity Tensors ... END");
}

} // namespace OpsFusion
