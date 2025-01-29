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

/*
 * mha_rope is a class to offload matrix
 * Attention masking and Softmax to AIE. this class uses lite runtime stack to
 * interface with XRT
 */
template <typename LhsT, typename TrigT, typename OutT>
class mha_rope : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_operand_header;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> default_shapes_;
  /* BxMxK dimension of base elwmul being offloaded to AIE */
  int64_t kernel_x_shape_[3];
  /*Kernel shape selected in runtime*/
  /* actual BxMxK of matrix A */
  int64_t operand_shape_[3];
  size_t operand_size_in_bytes_;
  size_t trig_size_in_bytes_;
  /* xrt context handle */
  // xrt_context *xrt_ctx_;
  // static instruction_registry instr_reg_add_;
  static std::once_flag instr_reg_flag_;
  static std::once_flag instr_reg_v1_flag_;
  /* XRT BO for tiled LHS matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled RHS matrix */
  xrt::bo b_bo_;
  /* XRT BO for tiled OUT matrix */
  xrt::bo c_bo_;
  /* size for activation dtype*/
  int operand_dtype_size_;
  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t b_copy_time_;
  int64_t b_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t mha_rope_id_;
  static uint64_t mha_rope_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string operand_dtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;
  std::string op_version_;
  std::string instr_bo_key_;
  std::vector<std::pair<size_t, size_t>> thresholds_;
  enum transpose_enum { NONE = 0, INPUT = 1, ALL = 2 };
  transpose_enum transpose_ = NONE;
  std::map<std::string, transpose_enum> transpose_attr{
      {"none", NONE}, {"input", INPUT}, {"all", ALL}};
  std::map<transpose_enum, std::string> transpose_txn_suffix{
      {NONE, ""}, {INPUT, "_input_trans"}, {ALL, "_trans"}};
  std::string model_ = "";
  std::map<std::string, std::string> model_string_attr{{"LLAMA2", ""},
                                                       {"CHATGLM", "_glm"}};
  /*
   * Utility function that setups for context.
   */
  void setup_instr_init();
  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry(const std::map<std::string, std::any> &attr);
  /*
   * Utility function that checks if an operands shape is supported before
   * execution.
   */
  bool isSupportedShape(const Tensor &operand);
  std::string get_instr_key(std::string prefix, size_t batch, size_t m,
                            size_t k) const;

public:
  void set_kernel_shape(std::vector<size_t> shape);
  mha_rope(const std::string &operand_dtype, bool load_xrt,
           const std::map<std::string, std::any> &attr = {});
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void execute(std::vector<xrt::bo> &input, std::vector<xrt::bo> &output,
               bool wait = true, int64_t offset = 0);
  void debug(bool enable);

  std::vector<xrt::bo> get_inputs();
  std::vector<xrt::bo> get_outputs();
  void set_params(const std::string &model_name,
                  std::vector<size_t> input_shape);

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override {}
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override {}
  // TBD
  inline static float EPSILON = 0;
  bool load_xrt_;
};

} // namespace ryzenai
