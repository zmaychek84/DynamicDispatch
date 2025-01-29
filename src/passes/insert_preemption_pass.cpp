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

#include <iostream>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <ops/pm_load/pm_load.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

static void insert_preemption_op_in_meta(Metadata &meta,
                                         const std::string &op_name,
                                         uint8_t pdi_id) {

  std::map<std::string, std::any> attr;
  // attr["op_name"] = op_name;
  Metadata::OpInfo preempt_info = {op_name, "PREEMPTION", {},   {},    {},
                                   {},      {},           attr, pdi_id};
  meta.op_list.emplace_back(preempt_info);
}

Metadata insert_preemption_nodes(const Metadata &meta) {
  RYZENAI_LOG_TRACE("Insert Preemption nodes: Init");
  Metadata preemption_meta = meta;
  preemption_meta.op_list.clear();

  // iterate over pdi partitions and insert profile points
  for (size_t part = 0; part < meta.partitions.size(); part++) {
    const auto &op2 = meta.op_list.at(meta.partitions.at(part).op_range.first);
    auto part_end = meta.partitions.at(part).op_range.second;
    for (size_t i = meta.partitions.at(part).op_range.first; i < part_end;
         ++i) {
      const auto &op = meta.op_list.at(i);
      // json op_prop = get_op_prop(meta, op);
      preemption_meta.op_list.emplace_back(op);
      // Skipping last preemption in the current partition.
      if (part_end - 1 != i) {
        insert_preemption_op_in_meta(preemption_meta,
                                     ("preemption_op_" + op.name), op.pdi_id);
      }
    }
  }

  RYZENAI_LOG_TRACE("Insert Preempt  nodes: Done");
  return preemption_meta;
}
} // namespace OpsFusion
