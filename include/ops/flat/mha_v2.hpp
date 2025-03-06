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

#include "ops/op_interface.hpp"
#include "ops/ops_common.hpp"

namespace ryzenai {
namespace flat {

template <typename InT, typename OutT> class mha_v2 : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<std::vector<size_t>>> default_shapes_;
  const uint64_t B_ = 1; // batch
  // query shape: (num_head, seq_len_q_, head_size)
  // key shape:   (num_head, head_size, seq_len_total_k)
  // value shape: (num_head, head_size, seq_len_total_k)
  // rope wts size: (seq_len_q, head_size)
  uint64_t seq_len_q_;
  uint64_t seq_len_total_k_;
  uint64_t num_heads_;
  uint64_t head_size_;
  size_t q_size_;
  size_t kv_size_;
  size_t mask_size_;
  size_t rope_wts_size_;
  size_t ifm_bo_size_;
  size_t total_k_bo_size_;
  size_t total_v_bo_size_;
  size_t ofm1_size_;
  size_t ofm2_size_;
  static std::once_flag instr_reg_flag_;
  xrt::bo ifm_bo_;
  xrt::bo total_k_bo_;
  xrt::bo total_v_bo_;
  xrt::bo ofm1_bo_;
  xrt::bo ofm2_bo_;
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  /* variables to store profile data */
  int64_t run_aie_time_ = 0;
  int64_t cpu_acc_time_ = 0;
  int64_t num_run_aie_ = 0;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t mha_v2_id_ = 0;
  static uint64_t mha_v2_count;
  /* debug flag */
  bool debug_ = false;
  bool skip_create_total_k_ = false;
  bool skip_create_total_v_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  std::string XCLBIN_FNAME_;
  std::string instr_bo_key_;
  void setup_instr_registry(const std::map<std::string, std::any> &attr);
  std::string get_instr_key(std::string prefix,
                            const std::vector<size_t> &mat) const;
  void set_kernel_shapes(const std::vector<size_t> &input_shape);

public:
  mha_v2(const std::string &a_dtype, const std::string &c_dtype, bool load_xrt,
         const std::map<std::string, std::any> &attr = {});
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void execute(std::vector<xrt::bo> &input, std::vector<xrt::bo> &output,
               size_t offset, bool wait = true);
  void debug(bool enable);
  xrt::bo create_bo(void *usr_ptr, size_t size, int operand_index);
  void set_params(const std::string &model_name,
                  const std::vector<size_t> &input_shape,
                  const std::map<std::string, std::any> &attr = {});

  void set_execute_kernel_shape(const std::vector<size_t> &input_shape);

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_transaction_bin() const;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  }
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;

  std::vector<xrt::bo> get_inputs();
  std::vector<xrt::bo> get_outputs();
};

} // namespace flat
} // namespace ryzenai
