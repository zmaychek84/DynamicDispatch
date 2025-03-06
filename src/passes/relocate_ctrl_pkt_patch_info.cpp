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

#include <iostream>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <utils/meta_utils.hpp>

#include "txn/txn_utils.hpp"
#include "utils/op_utils.hpp"

#include "passes.hpp"

namespace OpsFusion {

void relocate_ctrl_pkt_patch_info(Metadata &meta, bool elf_flow) {
  auto param_offset =
      elf_flow ? OpArgMap::CTRL_PKT_BIN : OpArgMap::CONST_KERNEL_PARAM_INPUT;
  for (size_t i = 0; i < meta.op_list.size(); i++) {
    auto &op_info = meta.op_list.at(i);
    auto args = OpsFusion::get_op_args(op_info);
    RYZENAI_LOG_TRACE(OpsFusion::dd_format("Get ctrl_pkt_patch_info for op:{}",
                                           op_info.name));
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    std::map<std::string, void *> const_buf_ptrs;
    std::vector<Tensor> tensors = MetaUtils::collect_op_tensors(meta, op_info);

    auto ctrl_pkts_patch_info =
        DD_INVOKE_OPMETHOD(get_ctrl_pkt_patch_info, op.get(), op_info, tensors,
                           tensors, op_info.attr);
    const auto args_map = DD_INVOKE_OPMETHOD(get_buffer_reqs, op.get(), op_info,
                                             tensors, tensors, op_info.attr);

    // update xrt arg idx based on the args map
    const auto argmap_partition =
        dynamic_dispatch::op_utils::partition_argmap(args_map);
    for (auto &patch : ctrl_pkts_patch_info) {
      const auto &op_arg = dynamic_dispatch::op_utils::find_op_arg(
          argmap_partition, patch.xrt_arg_idx, patch.bo_offset);
      if (op_arg.arg_type == OpArgMap::CONST_KERNEL_PARAM_INPUT) {
        auto tensor_offset = MAP_AT(meta.super_instr_map, op_info.name).offset;
        auto final_offset = patch.bo_offset + tensor_offset;
        auto orig_arg_idx = patch.xrt_arg_idx;
        patch.xrt_arg_idx = OpArgMap::CONST_KERNEL_PARAM_INPUT;
        patch.bo_offset = final_offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching super intr to: xrt_arg_idx:{} - > {}, bo_offset:{}, "
            "fused_bo_offset: {}",
            orig_arg_idx, patch.xrt_arg_idx, patch.bo_offset, tensor_offset));
      } else if (op_arg.arg_type == OpArgMap::CONST_INPUT) {
        auto tensor_offset = MAP_AT(meta.const_map, op_info.name).offset;
        auto final_offset = patch.bo_offset + tensor_offset;
        auto orig_arg_idx = patch.xrt_arg_idx;
        patch.xrt_arg_idx = OpArgMap::CONST_INPUT;
        patch.bo_offset = final_offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching const input to: xrt_arg_idx:{} -> {}, bo_offset:{}, "
            "fused_bo_offset: {}",
            orig_arg_idx, patch.xrt_arg_idx, patch.bo_offset, tensor_offset));
      } else if (op_arg.arg_type == OpArgMap::CTRL_PKT_BIN) {
        auto orig_arg_idx = patch.xrt_arg_idx;
        patch.xrt_arg_idx = param_offset;
        patch.bo_offset =
            patch.bo_offset + meta.ctrl_pkt_map.at(op_info.name).offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching ctrl_pkt_bin buffer to: xrt_arg_idx:{} -> {}, "
            "bo_offset:{}, fused_bo_offset: {}",
            orig_arg_idx, patch.xrt_arg_idx, patch.bo_offset,
            meta.ctrl_pkt_map.at(op_info.name).offset));
      } else if (op_arg.arg_type == OpArgMap::SCRATCH_PAD) {
        auto scratch_offset = MAP_AT(meta.fused_tensors, "scratch").size;
        auto final_offset = patch.bo_offset + scratch_offset;
        auto orig_arg_idx = patch.xrt_arg_idx;
        patch.xrt_arg_idx = OpArgMap::SCRATCH_PAD;
        patch.bo_offset = final_offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching scratch buffer to: xrt_arg_idx:{} -> {}, bo_offset:{}, "
            "fused_bo_offset: {}",
            orig_arg_idx, patch.xrt_arg_idx, patch.bo_offset, scratch_offset));
      } else if ((op_arg.arg_type == OpArgMap::INPUT) ||
                 (op_arg.arg_type == OpArgMap::OUTPUT)) {
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Op Arg: arg_type:{} idx: {}, onnx_arg_idx: {}, offset: {}",
            op_arg.arg_type, op_arg.xrt_arg_idx, op_arg.onnx_arg_idx,
            op_arg.offset));
        const size_t onnx_argidx = op_arg.onnx_arg_idx;
        const auto &arg_label = ARRAY_AT(args, onnx_argidx);
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Onnx Arg idx: {}, arg_label: {}", onnx_argidx, arg_label));
        const auto &tensor = MAP_AT(meta.tensor_map, arg_label);
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Tensor: arg_idx: {}, offset: {}, size: {}", tensor.arg_idx,
            tensor.offset, tensor.size_in_bytes));

        size_t new_argidx = tensor.arg_idx;
        size_t block_offset = tensor.offset;
        size_t curr_offset_delta = patch.bo_offset - op_arg.offset;
        // tensor.offset tells where data actually is
        // op_arg.padding_offset is op requirement on whether it needs address
        // of actual data or beginning of padding
        size_t final_offset =
            block_offset + curr_offset_delta - op_arg.padding_offset;
        auto orig_arg_idx = patch.xrt_arg_idx;
        patch.xrt_arg_idx = new_argidx;
        patch.bo_offset = final_offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching I/O bo to: xrt_arg_idx:{} -> {}, bo_offset:{}, "
            "fused_bo_offset: {}",
            orig_arg_idx, patch.xrt_arg_idx, patch.bo_offset, tensor.offset));
      } else {
        DD_THROW(dd_format("Unknown arg type for op {}", op_info.name));
      }
    }
    op_info.ctrl_pkt_patch_info = std::move(ctrl_pkts_patch_info);
  }
}
} // namespace OpsFusion
