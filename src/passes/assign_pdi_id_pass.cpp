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
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

void assign_pdi_id_pass(const OpPDIMap &op_pdi_map, Metadata &meta) {

  std::set<std::uint8_t> unique_pdi_ids;

  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    auto &op = meta.op_list.at(i);
    std::uint8_t pdi_id = OpsFusion::get_pdi_id(op_pdi_map, op.type);
    op.pdi_id = pdi_id;
    unique_pdi_ids.insert(pdi_id);
  }

  constexpr std::uint8_t DEFAULT_PDI_ID = 0;

  size_t num_unique_pdi_ids = unique_pdi_ids.size();

  // only have control ops
  // e.g. want to profile sequence of PM loads
  bool use_default_pdi_id =
      (num_unique_pdi_ids == 1) &&
      (unique_pdi_ids.end() != unique_pdi_ids.find(OpsFusion::CONTROL_PDI_ID));

  if (use_default_pdi_id) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format("Using Default PDI ID: {}",
                                           (std::uint32_t)DEFAULT_PDI_ID));
    for (size_t i = 0; i < meta.op_list.size(); ++i) {
      auto &op = meta.op_list.at(i);
      op.pdi_id = DEFAULT_PDI_ID;
    }
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Assign PDI IDs: DONE"));
}

} // namespace OpsFusion
