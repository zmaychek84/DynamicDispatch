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
// sd3.0
namespace sd {
template <typename InT, typename OutT> class slice : public OpInterface {
private:
  // [2x1280]
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<std::vector<size_t>>> default_shapes_;
  std::map<std::string, std::any> attr_;

  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> dimensions_;
  const size_t b_bo_size_ = 128;

  size_t const_input_size_;
  static std::once_flag instr_reg_flag_;

  /* XRT BO for input matrix */
  xrt::bo a_bo_;
  xrt::bo b_bo_;
  /* XRT BO for tiled OUT matrix */
  xrt::bo c_bo_;

  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t b_copy_time_;
  int64_t b_sync_time_;
  int64_t run_aie_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t slice_id_;
  static uint64_t slice_count;
  // /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/

  std::string ifm_dtype_;
  std::string ofm_dtype_;
  int ifm_dtype_size_;
  int ofm_dtype_size_;
  std::string txn_fname_prefix_;
  std::string XCLBIN_FNAME_;
  const std::string sd_slice_key_ = "sd_slice_";
  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix,
                            const std::vector<size_t> &dimensions) const;

public:
  slice(const std::string &ifm_dtype, const std::string &out_dtype,
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
  // const std::vector<uint8_t> get_transaction_bin() const;

  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;

  const std::map<std::string, std::any> &get_attr() const override {
    return attr_;
  }

  void set_params();

  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  };
};

} // namespace sd

} // namespace ryzenai
