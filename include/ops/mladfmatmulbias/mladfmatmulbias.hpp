// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
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

template <typename InT, typename WtT, typename AccT, typename OutT = AccT>
class mladfmatmulbias : public OpInterface {
private:
  // additional member variables grows from matmulbias
  /* use AVX or not */
  bool use_avx;
  /* bytes required for params */
  int params_bytes;
  /*group size selected for this instantiation */
  int grp_size_;
  /* singed or unsigned */
  int sign;
  /* Temporary CPU buffer to hold accumulation */
  std::vector<AccT> c_acc_vec_;

  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;

  std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<size_t, size_t>>>
      all_thresholds_;
  mutable std::vector<std::pair<size_t, size_t>> thresholds_;
  std::string DPU_DIR;
  /* M x K dimension of base matmul being offloaded to AIE */
  mutable int64_t kernel_x_shape_[2];
  /* K x N dimension of base matmul being offloaded to AIE */
  mutable int64_t kernel_y_shape_[2];
  /* M x N dimension of base matmul being offloaded to AIE */
  mutable int64_t kernel_z_shape_[2];
  /*Kernel shape selected in runtime*/
  int64_t kernel_x_rows;
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX;
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x N of matrix C */
  int64_t c_shape_[2];
  /* actual K x N of matrix W */
  mutable int64_t w_shape_[2];
  /* padded shape of weight matrix */
  int64_t w_padded_shape_[2];
  // static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  static std::once_flag instr_reg_v1_flag_;
  static std::once_flag supported_shapes_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* vector of XRT BOs for tiled and reformtted weight matrix */
  std::vector<xrt::bo> weights_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_token_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_token_;
  /* size for activation dtype */
  int a_dtype_size_;
  /* size for weights dtype*/
  int b_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  size_t max_a_bo_size_;
  size_t max_c_bo_size_;
  size_t max_m_;
  size_t max_k_;
  size_t max_n_;

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
  uint64_t mladfmatmulbias_id_;
  static uint64_t mladfmatmulbias_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  std::string op_version_;
  bool initialized_;
  /*tiling information*/
  std::map<int64_t, double> m_tiling_cost_;
  static std::mutex instr_reg_mutex_;
  bool input_realloc_;
  bool output_realloc_;

  std::vector<mladf_matrix_shapes> supported_shapes_;

  void setup_instr_init();
  void setup_instr_registry(const std::map<std::string, std::any> &attr = {});
  void setup_supported_shapes();

  std::string
  get_instr_key(std::string prefix, size_t m, size_t k, size_t n,
                size_t grp_size = 0 /* additional arg for group size*/) const;

  std::tuple<mladf_matrix_shapes, std::vector<int64_t>, double>
  map_padded_shape(int64_t M, int64_t K, int64_t N) const;
  const std::vector<uint8_t>
  generate_fused_txnbin(const mladf_matrix_shapes &tiling_info,
                        const std::vector<int64_t> tiling_info_m,
                        const int64_t &K, const int64_t &N,
                        const int64_t &group_size) const;

  void reformat_const(const std::vector<Tensor> &const_params,
                      const std::map<std::string, std::any> &attr,
                      std::vector<std::vector<std::uint8_t>> &const_vecs,
                      bool is_online);

public:
  mladfmatmulbias(const std::string &a_dtype, const std::string &b_dtype,
                  const std::string &c_dtype, bool load_xrt,
                  const std::map<std::string, std::any> &attr = {});
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  std::vector<std::vector<std::uint8_t>>
  export_const_params(const std::vector<Tensor> &const_params,
                      const std::map<std::string, std::any> &attr = {});
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;

  void execute(std::vector<xrt::bo> &input, std::vector<xrt::bo> &output,
               bool wait = true);
  void execute_2(std::vector<xrt::bo> &input, std::vector<xrt::bo> &output,
                 bool wait = true);
  void execute(std::vector<uint64_t> &input, std::vector<xrt::bo> &output,
               bool wait = true);
  void execute_2(std::vector<uint64_t> &input, std::vector<xrt::bo> &output,
                 bool wait = true);

  void execute_internal(std::vector<Tensor> &input_Tensor,
                        std::vector<Tensor> &output_Tensor, int wts_index,
                        bool wait = true);
  void execute_internal_2(std::vector<Tensor> &input_Tensor,
                          std::vector<Tensor> &output_Tensor, int wts_index,
                          bool wait = true);
  void execute_2(std::vector<Tensor> &input, std::vector<Tensor> &output);
  std::vector<xrt::bo> get_inputs(int M);
  std::vector<xrt::bo> get_outputs(int M);
  std::vector<xrt::bo> get_const();
  void set_shape(std::vector<size_t> a_shape, std::vector<size_t> wt_shape,
                 int group_size);
  void set_shape_2(std::vector<size_t> a_shape, std::vector<size_t> wt_shape,
                   int group_size);
  void debug(bool enable);
  std::vector<mladf_matrix_shapes> &get_supported_shapes();

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  void set_kernel_shapes_kn_mladf() const;
  // void initialize_weights_int4_mladf(const std::vector<Tensor>
  // &const_params);
  void set_kernel_shapes_m_mladf(int64_t input_m);
  void run_aie(InT *a, xrt::bo &w_bo, int64_t *input_shape, bool wait = true);
  void run_aie_2(InT *a, xrt::bo &w_bo, int64_t *input_shape, bool wait = true);
  bool create_bo(void *use_ptr, size_t size, int operand_index);
};

} // namespace ryzenai
