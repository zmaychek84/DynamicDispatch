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

namespace ryzenai {

struct overlay_pm_meta {
  struct pkt_switch_meta {
    uint8_t pkt_id;
    uint8_t col;
    uint8_t dma_ch_num;
  };
  uint8_t num_cols;
  std::vector<pkt_switch_meta> pkt_sw_meta_;
};

struct op_xclbin_meta {
  std::string xclbin_name;
  std::string pm_elf_fname;
};

class pm_load : public OpInterface {
private:
  static const std::map<std::string, overlay_pm_meta> overlay_meta_;
  static const std::map<std::string, op_xclbin_meta> op_xclbin_meta_;
  const std::vector<uint8_t>
  get_pm_bin(const std::map<std::string, std::any> &attr) const;

public:
  pm_load(bool load_xrt = false);
  void execute(std::string op_name, std::string dtype);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr) const override;
  const overlay_pm_meta &get_overlay_meta(const std::string &xclbin_name) const;
  const op_xclbin_meta &get_op_xclbin_meta(const std::string &op_name,
                                           const std::string &dtype) const;
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr) const override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr) const override;
  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override {}
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override {}
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override {}
};

} // namespace ryzenai
