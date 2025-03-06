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
template <typename InT, typename OutT> class concateOps : public OpInterface {
private:
  std::map<std::string, std::vector<matrix_shapes>> default_shapes_;
  int64_t graphId_;
  int64_t inChannels_;
  int64_t outChannels_;

  //  static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo ifmBo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo constBo_;
  /* XRT BO for tiled output matrix */
  xrt::bo ofmBo_;
  /* size for input activation dtype*/
  xrt::bo scratchBo_;
  /* size for scratch pad buffer*/

  /* variables to store profile data */
  int64_t run_aie_time_;

  static std::once_flag logger_flag_;
  uint64_t concatenate_id_;
  static uint64_t concatenate_count;
  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string weightDtype_;
  std::string ofmDtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;
  std::string model_variant_;

  std::vector<std::unique_ptr<OpInterface>> op_interfaces_;
  /* Add the CreateConvOperator function declaration */
  void CreateConvOperator(const std::map<std::string, std::any> &attrs);
  void CreateMaxpoolOperator(const std::map<std::string, std::any> &attrs);
  void CreateLstmOperator(const std::map<std::string, std::any> &attrs);
  void
  CreateConvForMatmulAddOperator(const std::map<std::string, std::any> &attrs);

  void setup_instr_registry();
  std::string get_instr_key(std::string prefix, int64_t graphId,
                            int64_t inChannels, int64_t outChannels) const;
  void WriteToFile(void *src, uint64_t length);

public:
  concateOps(const std::string &ifmDtype, const std::string &ofmDtype,
             bool load_xrt,
             const std::map<std::string, std::any> &attributes = {});
  void set_params(const std::string &modelName, bool debugFlag);
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  }
};

} // namespace ryzenai
