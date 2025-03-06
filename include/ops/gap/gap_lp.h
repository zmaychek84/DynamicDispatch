//
// Created by abhbisht on 9/13/2024.
//

#ifndef GAP_LP_H
#define GAP_LP_H
#include <cmath>
#include <string>
#include <vector>

namespace gap_lp {

struct LayerInfo {
  uint8_t div_shift = 0;

  uint16_t ifm_depth = 0;
  uint16_t ifm_width = 0;
  uint16_t ifm_height = 0;
  uint16_t ofm_depth = 0;
  uint16_t ofm_width = 0;
  uint16_t ofm_height = 0;
  uint16_t batch_size = 0;

  uint32_t div_factor = 0;

  uint64_t offset = 0;
};

struct TilingInfo {
  static TilingInfo getInstance(const std::string &);

  uint8_t n_adf_rows = 0;
  uint8_t n_adf_cols = 0;
  uint8_t ifm_sv_width = 0;
  uint8_t ifm_sv_width_eff = 0;
  uint8_t ifm_sv_height = 0;
  uint8_t ifm_sv_depth = 0;

  std::vector<uint8_t> super_iter;
};

class LayerParams {
  std::vector<uint8_t> layerParams;

  void set_8_bits(size_t idx, uint8_t val) { layerParams.at(idx) = val; }

  void set_16_bits(size_t idx, uint16_t val) {
    for (int i = 0; i < 2; i++) {
      layerParams.at(idx + i) = (val >> 8 * i) & 0xFF;
    }
  }

  void set_32_bits(size_t idx, uint32_t val) {
    for (int i = 0; i < 4; i++) {
      layerParams.at(idx + i) = (val >> 8 * i) & 0xFF;
    }
  }

  void set_64_bits(size_t idx, uint64_t val) {
    for (int i = 0; i < 8; i++) {
      layerParams.at(idx + i) = (val >> 8 * i) & 0xFF;
    }
  }
};

class LayerParamsWithQdq {
  constexpr static size_t N_BYTES_LP = 64;

  constexpr static size_t REP_COUNT_OFFSET = 8;
  constexpr static size_t OFFSET_17 = 17;
  constexpr static size_t OFFSET_18 = 18;
  constexpr static size_t OFFSET_19 = 19;
  constexpr static size_t OFFSET_22 = 22;
  constexpr static size_t OFFSET_23 = 23;

  constexpr static size_t OFFSET_OFFSET = 40;

  constexpr static size_t IFM_SV_WIDTH_OFFSET = 48;
  constexpr static size_t IFM_SV_WIDTH_EFF_OFFSET = 49;
  constexpr static size_t IFM_SV_HEIGHT_OFFSET = 50;
  constexpr static size_t IFM_SV_DEPTH_OFFSET = 51;

  constexpr static size_t DIV_SHIFT_OFFSET = 52;

  constexpr static size_t OFFSET_53 = 53;

  constexpr static size_t DIV_FACTOR_OFFSET = 54;

  std::vector<uint8_t> layerParams;

  void set_8_bits(size_t idx, uint8_t val) { layerParams.at(idx) = val; }

  void set_16_bits(size_t idx, uint16_t val) {
    for (int i = 0; i < 2; i++) {
      layerParams.at(idx + i) = (val >> 8 * i) & 0xFF;
    }
  }

  void set_32_bits(size_t idx, uint32_t val) {
    for (int i = 0; i < 4; i++) {
      layerParams.at(idx + i) = (val >> 8 * i) & 0xFF;
    }
  }

  void set_64_bits(size_t idx, uint64_t val) {
    for (int i = 0; i < 8; i++) {
      layerParams.at(idx + i) = (val >> 8 * i) & 0xFF;
    }
  }

public:
  LayerParamsWithQdq() { layerParams.resize(N_BYTES_LP, 0); }

  // 8 bits
  void set_rep_count(uint8_t val) { set_8_bits(REP_COUNT_OFFSET, val); }
  void set_val_17(uint8_t val) { set_8_bits(OFFSET_17, val); }
  void set_val_18(uint8_t val) { set_8_bits(OFFSET_18, val); }
  void set_val_19(uint8_t val) { set_8_bits(OFFSET_19, val); }
  void set_val_22(uint8_t val) { set_8_bits(OFFSET_22, val); }
  void set_val_23(uint8_t val) { set_8_bits(OFFSET_23, val); }
  void set_offset(uint64_t val) { set_64_bits(OFFSET_OFFSET, val); }
  void set_ifm_sv_height(uint8_t val) { set_8_bits(IFM_SV_HEIGHT_OFFSET, val); }
  void set_ifm_sv_width(uint8_t val) { set_8_bits(IFM_SV_WIDTH_OFFSET, val); }
  void set_ifm_sv_width_eff(uint8_t val) {
    set_8_bits(IFM_SV_WIDTH_EFF_OFFSET, val);
  }
  void set_ifm_sv_depth(uint8_t val) { set_8_bits(IFM_SV_DEPTH_OFFSET, val); }
  void set_div_shift(uint8_t val) { set_8_bits(DIV_SHIFT_OFFSET, val); }
  void set_val_53(uint8_t val) { set_8_bits(OFFSET_53, val); }
  void set_div_factor(uint32_t val) { set_32_bits(DIV_FACTOR_OFFSET, val); }

  std::vector<uint8_t> data() { return this->layerParams; }
};

std::string generateTilingKey(const LayerInfo &layerInfo);

std::vector<uint8_t> computeLayerParams(const LayerInfo &);
} // namespace gap_lp
#endif // GAP_LP_H
