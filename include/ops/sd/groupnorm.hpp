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
#include <map>
#include <tuple>
#include <utility>

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {
// stable diffusion
namespace sd {
template <typename InT, typename WtT, typename OutT>
class groupnorm : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<std::tuple<int, int, int, int>>>
      default_shapes_;

  std::string design_param_;
  /* actual B x H x W x C of matrix A */
  int64_t a_shape_[4];
  /* actual B x H x W x C of matrix C */
  int64_t c_shape_[4];

  int64_t wts_size_;
  size_t b_bo_size_;

  int a_dtype_size_;
  int b_dtype_size_;
  int c_dtype_size_;

  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo b_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t b_copy_time_;
  int64_t b_format_time_;
  int64_t b_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t groupnorm_id_;
  static uint64_t groupnorm_count;
  bool debug_ = false;
  const std::string sd_gn_key_ = "sd_groupnorm_";
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;

  std::string txn_fname_prefix_;

  std::string XCLBIN_FNAME_;
  std::string pdi_name_;

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, size_t b, size_t h, size_t w,
                            size_t c) const;

public:
  groupnorm(const std::string &a_dtype, const std::string &b_dtype,
            const std::string &c_dtype, bool load_xrt,
            const std::map<std::string, std::any> &attr);
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void debug(bool enable);
  void set_params(const std::string &xclbin, const std::string &pdi_name);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  }
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
};
} // namespace sd
} // namespace ryzenai
