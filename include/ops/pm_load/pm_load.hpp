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

namespace ryzenai {

struct overlay_pm_meta {
  struct pkt_switch_meta {
    uint8_t pkt_id;
    uint8_t col;
    uint8_t dma_ch_num;
    uint8_t num_cores;
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
  static OpsFusion::OverlayPMMeta overlay_meta_;
  static OpsFusion::OpPMMap op_pm_map_;

  const std::vector<uint8_t>
  get_pm_bin(const std::map<std::string, std::any> &attr) const;
  std::string PM_PREFIX;

public:
  pm_load(bool load_xrt = false);
  void update_meta(const OpsFusion::OpPMMap &op_pm_map,
                   const OpsFusion::OverlayPMMeta &overlay_meta);
  void execute(std::string op_name, std::string dtype);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr) const override;
  const OpsFusion::OpPMMap::PMBinMetaInfo
  get_pmbin_meta(const std::string &pm_bin_name) const;
  const std::string get_op_pmbin_name(const std::string &op_name,
                                      const std::string &dtype) const;
  const std::uint32_t get_pm_core_size(const std::string pm_name,
                                       const std::uint8_t c,
                                       const std::uint8_t r,
                                       const std::uint8_t num_cores) const;
  const std::uint32_t get_pm_core_offset(const std::string pm_name,
                                         const std::uint8_t c,
                                         const std::uint8_t r,
                                         const std::uint8_t num_cores) const;
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
