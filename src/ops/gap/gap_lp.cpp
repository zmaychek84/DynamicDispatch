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

#include "ops/gap/gap_lp.h"

#include <iostream>
#include <map>
#include <utils/tfuncs.hpp>

namespace gap_lp {

struct m3uec_Layer1 : TilingInfo {
  m3uec_Layer1() {
    this->n_adf_rows = 8;
    this->n_adf_cols = 1;
    this->ifm_sv_height = 1;
    this->ifm_sv_width = 64;
    this->ifm_sv_width_eff = 49;
    this->ifm_sv_depth = 128;
    this->super_iter = {2, 1};
  }
};

std::map<std::string, TilingInfo> TILING_INFO_MAP = {
    {"1_49_1024_1_1_1024", m3uec_Layer1()}};

TilingInfo TilingInfo::getInstance(const std::string &key) {
  if (TILING_INFO_MAP.find(key) != TILING_INFO_MAP.end()) {
    return TILING_INFO_MAP[key];
  }
  DD_THROW("Invalid Tiling Key : " + key);

  // to suppress warning
  return {};
}

std::string generateTilingKey(const LayerInfo &layerInfo) {
  std::string tiling_key;
  std::vector inputShape = {layerInfo.ifm_height, layerInfo.ifm_width,
                            layerInfo.ifm_depth};
  std::vector outputShape = {layerInfo.ofm_height, layerInfo.ofm_width,
                             layerInfo.ofm_depth};
  for (auto dim : inputShape) {
    tiling_key += std::to_string(dim) + "_";
  }
  for (auto dim : outputShape) {
    tiling_key += std::to_string(dim) + "_";
  }
  /* ignore last "_" */
  tiling_key = tiling_key.substr(0, tiling_key.size() - 1);
  // std::cout << "Tiling key : " << tiling_key << std::endl;
  return tiling_key;
}

std::vector<uint8_t> computeLayerParams(const LayerInfo &layerInfo) {
  auto tilingKey = generateTilingKey(layerInfo);
  auto tilingInfo = TilingInfo::getInstance(tilingKey);

  LayerParamsWithQdq layerParamsWithQdq;
  layerParamsWithQdq.set_rep_count(tilingInfo.super_iter.at(0));
  layerParamsWithQdq.set_ifm_sv_depth(tilingInfo.ifm_sv_depth);
  layerParamsWithQdq.set_ifm_sv_height(tilingInfo.ifm_sv_height);
  layerParamsWithQdq.set_ifm_sv_width(tilingInfo.ifm_sv_width);
  layerParamsWithQdq.set_ifm_sv_width_eff(tilingInfo.ifm_sv_width_eff);

  layerParamsWithQdq.set_val_17(1);
  layerParamsWithQdq.set_val_18(1);
  layerParamsWithQdq.set_val_19(10);
  layerParamsWithQdq.set_val_22(1);
  layerParamsWithQdq.set_val_23(1);
  layerParamsWithQdq.set_val_53(8);

  layerParamsWithQdq.set_offset(layerInfo.offset);
  layerParamsWithQdq.set_div_shift(layerInfo.div_shift);
  layerParamsWithQdq.set_div_factor(layerInfo.div_factor);

  return layerParamsWithQdq.data();
}

} // namespace gap_lp
