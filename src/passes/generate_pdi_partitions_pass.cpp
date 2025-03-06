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
#include <unordered_set>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <utils/meta_utils.hpp>
#include <utils/pass_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

using op_type_t = std::string;

static bool is_cpu(const CPUOpList &cpu_ops, op_type_t op_type) {
  if (cpu_ops.find(op_type) != cpu_ops.end()) {
    return true;
  } else {
    return false;
  }
}

void generate_pdi_partitions_pass(Metadata &meta, bool eager_mode,
                                  const CPUOpList &cpu_ops) {

  std::vector<Partition> partitions;

  if (0 == meta.op_list.size()) {
    meta.partitions = partitions;
    return;
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Generate PDI Partitions Pass, eager_mode {}",
                           static_cast<std::uint32_t>(eager_mode)));

  std::set<std::uint8_t> unique_pdi_ids;

  Partition partition;

  size_t start_op_id = 0;
  auto curr_pdi_id = meta.op_list.at(0).pdi_id;
  bool curr_is_cpu = is_cpu(cpu_ops, meta.op_list.at(0).type);
  partition.pdi_id = curr_pdi_id;
  partition.is_cpu = curr_is_cpu;
  unique_pdi_ids.insert(curr_pdi_id);

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("\top_name {} pdi_id {}",
                                         meta.op_list[0].name,
                                         (std::uint32_t)partition.pdi_id));

  for (size_t op_id = 1; op_id < meta.op_list.size(); op_id++) {
    curr_pdi_id = meta.op_list.at(op_id).pdi_id;
    curr_is_cpu = is_cpu(cpu_ops, meta.op_list.at(op_id).type);
    RYZENAI_LOG_TRACE(OpsFusion::dd_format("\top_name {} pdi_id {}",
                                           meta.op_list[op_id].name,
                                           (std::uint32_t)curr_pdi_id));

    if ((partition.pdi_id != curr_pdi_id) ||
        (partition.is_cpu != curr_is_cpu) || eager_mode) {
      partition.op_range = std::make_pair(start_op_id, op_id);
      partitions.push_back(partition);

      start_op_id = op_id;
      partition.pdi_id = curr_pdi_id;
      partition.is_cpu = curr_is_cpu;
      unique_pdi_ids.insert(curr_pdi_id);
    }
  }

  partition.op_range = std::make_pair(start_op_id, meta.op_list.size());
  partitions.push_back(partition);

  if (unique_pdi_ids.end() != unique_pdi_ids.find(OpsFusion::CONTROL_PDI_ID)) {
    DD_THROW(OpsFusion::dd_format(
        "Found CONTROL_PDI_ID - this does not belong to any kernel!"));
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Num PDI Partitions {} : ", partitions.size()));
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Generate PDI partitions: DONE"));

  dynamic_dispatch::pass_utils::link_npu_partitions(partitions);
  meta.partitions = std::move(partitions);
}

} // namespace OpsFusion
