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

#include <op_fuser/fuse_types.hpp>
#include <op_fuser/fusion_rt.hpp>

namespace OpsFusion {

void assign_pdi_id_pass(const OpPDIMap &op_pdi_map, Metadata &meta);
Metadata insert_pm_swap_nodes(const Metadata &meta);
Metadata insert_record_timer_nodes(const Metadata &meta,
                                   uint32_t profile_level);
Metadata insert_preemption_nodes(const Metadata &meta);
std::pair<std::map<std::string, std::any>, std::map<std::string, std::any>>
get_record_timer_attr(const std::string &op_name);
void generate_pdi_partitions_pass(Metadata &meta, bool eager_mode);
void analyze_buffer_reqs(Metadata &meta);
void optimize_scratch_buffer(Metadata &meta);
bool split_max_partition_pass(
    Metadata &meta, const std::vector<std::vector<uint8_t>> fused_instr_vec,
    size_t limit);
void fetch_op_txn_bins(Metadata &meta,
                       std::map<std::string, SimpleSpan> &const_map,
                       bool elf_flow = false);
void relocate_ctrl_pkt_patch_info(Metadata &meta, bool elf_flow = false);

} // namespace OpsFusion
