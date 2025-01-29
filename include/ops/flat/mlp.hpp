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

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {
namespace flat {
template <typename InT, typename WtT, typename OutT>
class mlp : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<std::vector<size_t>>> default_shapes_;
  // ifm (M, K)
  // wts (K, N)
  // ofm (M, N)
  uint64_t M_;
  uint64_t K_;
  uint64_t N_;
  size_t ifm_size_;
  size_t wts_size_;
  size_t ofm_size_;
  uint64_t sv_k_num_;
  uint64_t sv_n_num_;
  uint64_t total_sv_k_num_;
  // TODO: make sv_k, sv_n configurable
  const size_t sv_k = 128;
  const size_t sv_n = 64;
  const size_t AIE_ALIGN = 64;
  size_t quants_sv_size_ = sv_k * sv_n / 2;
  size_t zp_sv_size_ =
      ((sv_k * sv_n / (2 * sv_k) + AIE_ALIGN - 1) / AIE_ALIGN) *
      AIE_ALIGN; // only sv_n/2, but align to AIE_ALIGN
  size_t scale_sv_size_ = sv_n * 2;
  size_t wts_vec_size_ = quants_sv_size_ + zp_sv_size_ + scale_sv_size_;

  size_t wts_bo_size_;
  size_t scale_bo_size_;
  size_t zp_bo_size_;
  size_t bias_bo_size_;

  static std::once_flag instr_reg_flag_;
  xrt::bo a_bo_; // ifm bo
  xrt::bo b_bo_; // const bo
  xrt::bo c_bo_; // ofm bo
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for weights dtype*/
  int b_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  /* variables to store profile data */
  int64_t run_aie_time_ = 0;
  int64_t cpu_acc_time_ = 0;
  int64_t num_run_aie_ = 0;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t mlp_id_ = 0;
  static uint64_t mlp_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  std::string XCLBIN_FNAME_;

  void setup_instr_registry();
  std::string get_instr_key(std::string prefix,
                            const std::vector<size_t> &mat) const;

public:
  mlp(const std::string &ifm_dtype = "bfloat16",
      const std::string &gate_wts_dtype = "uint8",
      const std::string &ofm_dtype = "bfloat16", bool load_xrt = false,
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
               bool wait = true);
  void debug(bool enable);
  bool create_bo(void *usr_ptr, size_t size, int operand_index);

  std::vector<xrt::bo> get_inputs();
  std::vector<xrt::bo> get_outputs();
  void set_params(std::vector<uint64_t> &input_shape, bool create_bo = true);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_transaction_bin() const;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  };
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  void cal_shuffled_wts_size(int64_t N, int64_t K);
  void wts_shuffle(std::vector<uint8_t> &bo_map, uint8_t *gate_weights,
                   uint8_t *gate_zp, float *gate_scales, float *gate_bias,
                   uint8_t *up_weights, uint8_t *up_zp, float *up_scales,
                   float *up_bias);
};
} // namespace flat
} // namespace ryzenai
