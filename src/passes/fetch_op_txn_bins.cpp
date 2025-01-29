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

#include "txn/txn_utils.hpp"
#include <op_fuser/fuse_types.hpp>
#include <op_fuser/fusion_rt.hpp>
#include <ops/op_builder.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

void fetch_op_txn_bins(Metadata &meta,
                       std::map<std::string, SimpleSpan> &const_map,
                       bool elf_flow) {
  for (size_t i = 0; i < meta.op_list.size(); i++) {
    auto &op_info = meta.op_list.at(i);
    op_info.attr["elf_flow"] = (uint32_t)elf_flow;
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Get ops txn for op:{}", op_info.name));
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);

    std::map<std::string, void *> const_buf_ptrs;
    for (auto &arg : op_info.const_args) {
      const_buf_ptrs[arg] = MAP_AT(const_map, arg).loc;
    }

    std::vector<Tensor> tensors =
        MetaUtils::collect_op_tensors(meta, op_info, const_buf_ptrs);

    auto txn_vec = DD_INVOKE_OPMETHOD(get_transaction_bin, op.get(), op_info,
                                      tensors, tensors, op_info.attr);
    auto args_map = DD_INVOKE_OPMETHOD(get_buffer_reqs, op.get(), op_info,
                                       tensors, tensors, op_info.attr);
    utils::txn_util patched_txn(txn_vec);
    patched_txn.patch(op_info, meta, args_map);
    op_info.txn_bin = std::move(patched_txn.to_vector());
  }
}
} // namespace OpsFusion
