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

template <typename InT, typename OutT> class nni_resize : public OpInterface {
private:
  std::string design_param_;
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_c_header;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> default_shapes_;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> raw_shapes_;
  /* H x W x C dimension of base nni_resize being offloaded to AIE */
  int64_t kernel_x_shape_[3];
  /*Kernel shape selected in runtime*/
  /* actual H x W x C of matrix A */
  int64_t a_shape_[3];
  /* actual H x W x C of matrix C */
  int64_t c_shape_[3];
  // static instruction_registry instr_reg_add_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t nni_resize_id_;
  static uint64_t nni_resize_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, size_t h, size_t w,
                            size_t c) const;
  std::tuple<size_t, size_t, size_t> map_padded_shape(size_t H, size_t W,
                                                      size_t C) const;

public:
  nni_resize(const std::string &a_dtype, const std::string &c_dtype,
             bool load_xrt, const std::map<std::string, std::any> &attr);

  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override {}
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void debug(bool enable);

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
};

} // namespace ryzenai
