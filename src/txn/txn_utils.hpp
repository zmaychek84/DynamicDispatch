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

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xaiengine.h>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_interface.hpp>

namespace utils {

class txn_util {
public:
  enum device_t {
    RYZENAI_PHX = XAIE_DEV_GEN_AIE2IPU,
    RYZENAI_STX = XAIE_DEV_GEN_AIE2P,
  };
  txn_util() = default;
  txn_util(const std::vector<uint8_t> &txn_vec);
  std::string summarize();
  std::string text_dump();
  static std::vector<uint8_t>
  patch(const std::vector<uint8_t> &txn,
        const std::vector<OpArgMap> &source_args_map,
        const std::vector<OpArgMap> &dest_args_map);
  static std::vector<uint8_t> patch(const std::vector<uint8_t> &txn,
                                    const std::vector<OpArgMap> &args_map);
  void patch(const OpsFusion::Metadata::OpInfo &op_info,
             const OpsFusion::Metadata &meta,
             const std::vector<OpArgMap> &args_map);
  std::vector<uint8_t> to_vector();
  static std::vector<uint8_t>
  fuse_txns(const std::vector<std::vector<uint8_t>> &txns);
  static void pass_through(uint8_t **ptr);
  static std::vector<std::uint8_t>
  convert_mc_code(const std::vector<std::uint8_t> &mc_code,
                  device_t device = RYZENAI_STX, std::uint32_t num_cols = 4);
  static void
  pactch_mc_control_packet(uint32_t *mc_control_packet,
                           size_t mc_control_packet_size, uint64_t ddr_base_ifm,
                           uint64_t ddr_base_param, uint64_t ddr_base_ofm,
                           uint64_t ddr_base_inter, bool pad_control_packet);

  std::vector<uint8_t> convert_to_opt_txn(const std::vector<uint8_t> &base_txn);

  void append_to_txn(XAie_DevInst *DevInst, uint8_t **ptr);
  std::vector<uint8_t> txn;

private:
  std::stringstream ss_hdr_;
  std::stringstream ss_ops_;
  std::stringstream ss_summary_;
  uint64_t txn_size_;
  uint64_t num_txn_ops_;
  uint64_t fused_size_;
  uint64_t fused_ops_;

  uint32_t num_w_ops = 0;
  uint32_t num_bw_ops = 0;
  uint32_t num_mw_ops = 0;
  uint32_t num_mp_ops = 0;
  uint32_t num_tct_ops = 0;
  uint32_t num_patch_ops = 0;
  uint32_t num_read_ops = 0;
  uint32_t num_readtimer_ops = 0;
  uint32_t num_mergesync_ops = 0;

  void stringify_txn_ops();
  void stringify_w32(uint8_t **ptr);
  void stringify_bw32(uint8_t **ptr);
  void stringify_mw32(uint8_t **ptr);
  void stringify_mp32(uint8_t **ptr);
  void stringify_tct(uint8_t **ptr);
  void stringify_patchop(uint8_t **ptr);
  void stringify_rdreg(uint8_t **ptr);
  void stringify_rectimer(uint8_t **ptr);
  void stringify_mergesync(uint8_t **ptr);
  void stringify_txn_bin();
  void prepare_summary();
};

} // namespace utils

class transaction_op {
public:
  /**
   * @brief create txn op created by aie_controller locally.
   * Format :
   *     | TRANSACTION_OP | SIZE | txn |
   */
  transaction_op(const std::vector<uint8_t> &txn);
  std::vector<uint8_t> get_txn_op();
  size_t get_txn_instr_size();

  static size_t getInstrBufSize(const std::string &txn_str);
  static void addTxnOp(const std::string &txn_str, void *instr_buf);

  // size of txn op header in bytes
  // this is the wrapper header around txn format supported by aie-rt
  constexpr static size_t TXN_OP_SIZE = 8;
  constexpr static uint32_t TXN_OP_CODE = 0;

private:
  std::vector<uint8_t> txn_op_;
};
