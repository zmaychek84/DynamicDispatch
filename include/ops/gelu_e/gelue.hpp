// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>
#include <ops/unary/unary.hpp>

namespace ryzenai {

/*
 * Gelu is a class to offload matrix
 * Gelu to AIE. this class uses lite runtime stack to interface with
 * XRT
 */
template <typename InT, typename OutT> class gelue : public unary<InT, OutT> {

public:
  gelue(const std::string &operand_dtype, bool load_xrt)
      : unary<InT, OutT>("gelue", operand_dtype, load_xrt) {
    unary<InT, OutT>::set_xclbinname(
        "xclbin/stx/gelue_4x4_abfloat16cbfloat.xclbin");
  };
};

} // namespace ryzenai
