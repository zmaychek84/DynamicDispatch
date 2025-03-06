
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

Metadata remove_identity_ops(const Metadata &meta) {
  Metadata identity_meta = meta;

  auto it = identity_meta.op_list.begin();
  while (it != identity_meta.op_list.end()) {
    if (it->type == "Identity") {
      auto args = OpsFusion::get_op_args(*it);
      // For an Identity op, args[0] is the input, args[1] is the output tensor
      // name.
      auto &input_tensor = MAP_AT(identity_meta.tensor_map, args[0]);
      auto &output_tensor = MAP_AT(identity_meta.tensor_map, args[1]);

      // Decide which tensor is the source and which one should be
      // updated. If the input tensor comes from "in"
      // update output_tensor, otherwise, update input_tensor.
      bool from_input = (input_tensor.parent_name == "in");
      auto &src = from_input ? input_tensor : output_tensor;
      auto &dest = from_input ? output_tensor : input_tensor;
      const std::string replace_arg = from_input ? args[1] : args[0];
      const std::string with_arg = from_input ? args[0] : args[1];

      // Propagate offset and parent properties.
      dest.arg_idx = src.arg_idx;
      dest.parent_name = src.parent_name;
      dest.offset = src.offset;

      // Update corresponding references in other operations.
      for (auto &other_op : identity_meta.op_list) {
        auto &args_to_update =
            from_input ? other_op.in_args : other_op.out_args;
        for (auto &arg : args_to_update) {
          if (arg == replace_arg) {
            arg = with_arg;
          }
        }
      }

      // Remove the tensor name from packed_tensors in fused_tensors.
      for (auto &fused_tensor : identity_meta.fused_tensors) {
        auto &packed_tensors = fused_tensor.second.packed_tensors;
        packed_tensors.erase(std::remove(packed_tensors.begin(),
                                         packed_tensors.end(), replace_arg),
                             packed_tensors.end());
      }

      // Remove the tensor entry which is no longer referenced.
      identity_meta.tensor_map.erase(replace_arg);
      it = identity_meta.op_list.erase(it);
    } else {
      ++it;
    }
  }
  return identity_meta;
}

void set_new_offsets(Metadata &org_meta, const Metadata &new_meta) {
  for (auto &tensor_pair : org_meta.tensor_map) {
    const auto &tensor_name = tensor_pair.first;
    if (new_meta.tensor_map.find(tensor_name) != new_meta.tensor_map.end()) {
      tensor_pair.second.offset = new_meta.tensor_map.at(tensor_name).offset;
    }
  }

  auto &scratch_buffer_org = MAP_AT(org_meta.fused_tensors, "scratch");
  auto &scratch_buffer_new = MAP_AT(new_meta.fused_tensors, "scratch");

  scratch_buffer_org.size = scratch_buffer_new.size;
}

} // namespace OpsFusion
