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

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {
// stable diffusion 1.5
namespace sd {
template <typename InT, typename WtT, typename BiasT, typename OutT>
class gemm : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<std::vector<size_t>>> default_shapes_;
  std::vector<size_t> curr_txn_shape_;

  /* actual input matrix */
  std::vector<int64_t> inputShape_;
  std::vector<int64_t> outputShape_;

  /* actual weight matrix inserted */
  int64_t weightShape_[2];
  int64_t K_;
  int64_t N_;
  bool bias_en_;
  // // will remove in the near future
  int sv_k_;
  int sv_n_;
  size_t CONST_BO_SIZE_;
  size_t IFM_BO_SIZE_;
  size_t OFM_BO_SIZE_;
  /* actual bias matrix inserted */

  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo ifmBo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo constBo_;
  /* XRT BO for tiled output matrix */
  xrt::bo ofmBo_;
  std::string instr_key_;

  int ifmDtypeSize_;
  /* size for weights dtype*/
  int weightDtypeSize_;
  /* size for bias dtype*/
  int biasDtypeSize_;
  /* size for output activation dtype*/
  int ofmDtypeSize_;
  /* variables to store profile data */
  int64_t ifmCopyTime_;
  int64_t ifmSyncTime_;
  int64_t weightCopyTime_;
  int64_t weightFormatTime_;
  int64_t weightSyncTime_;
  int64_t ofmCopyTime_;
  int64_t ofmSyncTime_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t gemm_id_;
  static uint64_t gemm_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string weightDtype_;
  std::string biasDtype_;
  std::string ofmDtype_;
  const std::string sd_gemm_key_ = "sd_gemm_";
  std::string txn_fname_prefix_;
  std::string XCLBIN_FNAME_;
  std::string pdi_name_;

  void setup_instr_registry();
  std::string get_key(std::string prefix, const std::vector<size_t> &mat) const;
  size_t get_const_bo_size(int sv_k, int sv_n) const;

public:
  gemm(const std::string &ifm_dtype, const std::string &weight_dtype,
       const std::string &bias_dtype, const std::string &out_dtype,
       bool load_xrt, const std::map<std::string, std::any> &attr = {});
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void debug(bool enable);
  void set_params(const std::string &xclbin, const std::string &pdi_name);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_transaction_bin() const;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  };
  const std::vector<uint8_t> get_super_kernel_params() const;
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  std::vector<uint8_t> shuffle_wts_bfp16(float *wts, float *bias);
};
} // namespace sd
} // namespace ryzenai
