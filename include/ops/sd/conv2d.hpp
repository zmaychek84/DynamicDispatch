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
#include <utils/utils.hpp>

namespace ryzenai {

// stable diffusion 1.5
namespace sd {

// add a 4th template parameter for bias type if necessary
template <typename InT, typename WtT, typename BiasT, typename OutT>
class conv : public OpInterface {
  using WtsListType = std::vector<std::vector<std::vector<std::vector<WtT>>>>;

private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<sd_conv2d_shapes>> default_shapes_;

  std::map<std::string, std::any> attr_;

  /* Input dimension of base conv being offloaded to AIE */
  int64_t kernelInputShape_[4];
  /* Weight dimension of base conv being offloaded to AIE */
  int64_t kernelWeightShape_[6];
  /* Weight dimension of base conv being offloaded to AIE */
  int64_t kernelBiasShape_[3];
  /* Output dimension of base conv being offloaded to AIE */
  int64_t kernelOutputShape_[4];

  /* actual input matrix */
  int64_t inputShape_[3];
  /* actual output matrix */
  int64_t outputShape_[3];
  int64_t outputShapeAligned_[3];
  /* actual weight matrix inserted */
  int64_t weightShape_[4];
  /* actual bias matrix inserted */
  int64_t biasShape_[1];

  int64_t N_;
  int64_t OC_; // output channel
  int64_t IC_; // input channel
  int64_t IH_; // input height
  int64_t IW_; // input width
  int64_t OH_; // output height
  int64_t OW_; // output width
  int64_t kh_; // kernel height
  int64_t kw_; // kernel width
  int64_t stride_;
  int64_t aligned_OC_;
  int64_t aligned_IC_;

  //  static instruction_registry instr_reg_;
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
  uint64_t conv_id_;
  static uint64_t conv_count;
  /* debug flag */
  bool debug_ = false;
  /* Add bias to output */
  bool bias_en_ = true;
  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string weightDtype_;
  std::string biasDtype_;
  std::string ofmDtype_;
  const std::string sd_conv_key_ = "sd_conv2d_";
  const int ic_min_sub_ = 8;
  const int oc_min_sub_ = 16;
  std::string txn_fname_prefix_;
  std::string XCLBIN_FNAME_;
  int batch_;
  std::string pdi_name_;

  void setup_instr_registry();
  std::string get_key(std::string prefix, int64_t OC, int64_t IC, int64_t IH,
                      int64_t IW, int64_t OH, int64_t OW, int64_t kh,
                      int64_t kw) const;
  size_t get_const_bo_size(uint32_t ifm_sv_depth) const;

public:
  conv(const std::string &ifm_dtype, const std::string &weight_dtype,
       const std::string &bias_dtype, const std::string &out_dtype,
       bool load_xrt, const std::map<std::string, std::any> &attr = {});
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void debug(bool enable);
  const std::vector<uint8_t> get_transaction_bin() const;
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::map<std::string, std::any> &get_attr() const override {
    return attr_;
  }
  void set_params(const std::string &xclbin, const std::string &pdi_name,
                  const sd_conv2d_shapes &shape_info);
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;

  const std::vector<uint8_t> get_super_kernel_params() const;
  void format_output(const Tensor &out_tensor, void *hw_out_ptr, size_t sz,
                     size_t tensor_idx,
                     const std::map<std::string, std::any> &attr = {}) override;
};

} // namespace sd

} // namespace ryzenai
