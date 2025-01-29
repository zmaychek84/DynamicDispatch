/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {
// stable diffusion 1.5
namespace sd {
template <typename InT, typename WtT, typename OutT>
class gelu : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> default_shapes_;

  /* actual B x M x N of matrix A */
  int64_t a_shape_[3];
  /* actual B x M x N of matrix A */
  int64_t c_shape_[3];

  int64_t B_;
  int64_t M_;
  int64_t N_;
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for weights dtype*/
  int b_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo b_bo_;
  const size_t b_bo_size_ = 128;
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
  uint64_t gelu_id_;
  static uint64_t gelu_count;
  /* debug flag */
  bool debug_ = false;
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  const std::string sd_gelu_key_ = "sd_gelu_";
  std::string txn_fname_prefix_;
  std::string XCLBIN_FNAME_;
  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, size_t b, size_t m,
                            size_t k) const;
  // we donot need pad in sd1.5
  //   std::tuple<size_t, size_t> map_padded_shape(size_t M, size_t N) const;

public:
  gelu(const std::string &a_dtype, const std::string &b_dtype,
       const std::string &c_dtype, bool load_xrt,
       const std::map<std::string, std::any> &attr);
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void debug(bool enable);
  void set_params();
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  };
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
};
} // namespace sd
} // namespace ryzenai
