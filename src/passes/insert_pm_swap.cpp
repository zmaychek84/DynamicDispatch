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
#include <ops/pm_load/pm_load.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

Metadata insert_pm_swap_nodes(const Metadata &meta, const OpPMMap &op_pm_map,
                              const OverlayPMMeta &overlay_pm_meta) {
  Metadata pm_swap_meta = meta;
  // clear op_list and rebuild the list by iterating through meta.
  pm_swap_meta.op_list.clear();
  std::string prev_pm_fname = "";
  uint32_t pm_id = 0;
  constexpr bool load_xrt = false;
  ryzenai::pm_load pm_op(load_xrt);
  pm_op.update_meta(op_pm_map, overlay_pm_meta);
  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    const auto &op = meta.op_list.at(i);
    auto args = OpsFusion::get_op_args(op);
    auto &op_type = op.type;
    auto &op_dtype = meta.tensor_map.at(args[0]).dtype;
    auto curr_pm_fname = pm_op.get_op_pmbin_name(op_type, op_dtype);
    if (prev_pm_fname != curr_pm_fname) {
      RYZENAI_LOG_TRACE(
          OpsFusion::dd_format("OP: {}, PM ID change from {} to {}", op.type,
                               prev_pm_fname, curr_pm_fname));
      prev_pm_fname = curr_pm_fname;

      std::map<std::string, std::any> attr;
      attr["op_type"] = op_type;
      attr["op_dtype"] = op_dtype;
      attr["pm_id"] = pm_id++;
      Metadata::OpInfo pm_op_info = {
          "pm_load_" + op.name, "PM_LOAD", {}, {}, {}, {}, {}, attr, op.pdi_id};
      pm_swap_meta.op_list.emplace_back(pm_op_info);
    }
    RYZENAI_LOG_INFO(
        OpsFusion::dd_format("OP: {}, PM ID: {}", op.type, curr_pm_fname));
    pm_swap_meta.op_list.emplace_back(op);
  }

  return pm_swap_meta;
}
} // namespace OpsFusion
