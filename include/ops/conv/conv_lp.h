// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef LP_COMPUTATION_H
#define LP_COMPUTATION_H
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace conv_lp {
constexpr static size_t aie_iw_aligned_size = 8;
constexpr static size_t aie_ic_aligned_size = 8;
constexpr static size_t DD_bytes = 64;
constexpr static size_t LP_bytes = 64;

class KernelControl {
  constexpr static uint32_t MAX = 0xFFFFFFFFu;
  constexpr static uint32_t ZERO_INIT_IDX = 0;
  constexpr static uint32_t SIGN_N_IDX = 1;
  constexpr static uint32_t SIGN_O_IDX = 2;
  constexpr static uint32_t SKIP_CASC_IN_IDX = 6;
  constexpr static uint32_t SKIP_CASC_OUT_IDX = 7;
  constexpr static uint32_t SIGN_W_IDX = 8;
  constexpr static uint32_t SIGN_A_IDX = 9;

  constexpr static uint32_t NORM_CH_G_IDX = 19;
  constexpr static uint32_t NORM_CH_G_SIZE = 8;
  uint32_t kernelControl;

public:
  KernelControl() { kernelControl = 0; }

  void set(uint8_t idx) { kernelControl |= (1 << idx); }

  void reset(uint8_t idx) { kernelControl &= (~1 << idx); }

  void set_zero_init(bool val) {
    val ? set(ZERO_INIT_IDX) : reset(ZERO_INIT_IDX);
  }
  void set_sign_N(bool val) { val ? set(SIGN_N_IDX) : reset(SIGN_N_IDX); }
  void set_sign_O(bool val) { val ? set(SIGN_O_IDX) : reset(SIGN_O_IDX); }
  void set_skip_casc_in(bool val) {
    val ? set(SKIP_CASC_IN_IDX) : reset(SKIP_CASC_IN_IDX);
  }
  void set_skip_casc_out(bool val) {
    val ? set(SKIP_CASC_OUT_IDX) : reset(SKIP_CASC_OUT_IDX);
  }
  void set_sign_W(bool val) { val ? set(SIGN_W_IDX) : reset(SIGN_W_IDX); }
  void set_sign_A(bool val) { val ? set(SIGN_A_IDX) : reset(SIGN_A_IDX); }
  void set_norm_ch_g(uint8_t val) {
    kernelControl |= (MAX & (val << NORM_CH_G_IDX));
  }
  void set_reserved_3(uint8_t val) { kernelControl |= 0x08 & val; }

  uint32_t data() { return kernelControl; }
};

struct LayerInfo {
  int32_t c1;
  int32_t c2;
  int32_t shift_conv;

  // TODO: check its data type correctness
  int32_t shift_out;

  // int layer_num;
};

struct TilingInfo {
  static TilingInfo getInstance(const std::string &);
  std::string type = "conv";
  uint16_t kernel_size = 0;
  std::vector<int> stride;
  std::vector<int> padding;
  uint32_t pad_value = 0;
  std::vector<int> dilation;
  std::string ifm_type = "uint16";
  std::string wgt_type = "uint16";
  std::string ofm_type = "uint16";

  // parallelism
  uint16_t oh = 1;
  uint16_t ow = 8;
  uint16_t ic = 8;
  uint16_t oc = 8;
  std::string activation;

  // schedule
  std::string layer_name = "l16";
  std::vector<int> align_ifm;
  std::vector<int> align_wts;
  std::vector<int> ofm_shape;
  std::vector<int> align_ofm_shape;
  std::vector<int> ofmsv_chunk;
  std::vector<int> ifmsv_chunk;
  uint16_t width_iter = 25;
  uint16_t depth_iter = 4;
  uint16_t height_iter = 1;
  uint16_t channel_iter = 5;
  std::vector<int> super_iter;

  std::vector<int> pool_ksize = {};
  std::vector<int> pool_strides = {};
};

class LayerParams {
  constexpr static size_t N_DD_PARM_BYTES = 64;

  constexpr static size_t SV_IW_OFFSET = 0;
  constexpr static size_t SV_IH_OFFSET = 1;
  constexpr static size_t ICG_OFFSET = 2;
  constexpr static size_t OCG_OFFSET = 3;
  constexpr static size_t K_Y_OFFSET = 4;
  constexpr static size_t LOG2STRIDE_OFFSET = 5;
  constexpr static size_t LET_ADD_PARAMS_OFFSET = 6;
  constexpr static size_t OUT_MODE_OFFSET = 7;
  constexpr static size_t SRS_SHIFT_5_OFFSET = 8;
  constexpr static size_t SRS_SHIFT_6_OFFSET = 9;
  constexpr static size_t SRS_SHIFT_0_OFFSET = 10;
  constexpr static size_t P_SUM_BUF_OFFSET_OFFSET = 11;
  constexpr static size_t LRELU_ALPHA_SCALED_1_OFFSET = 12;
  constexpr static size_t LRELU_ALPHA_SCALED_2_OFFSET = 13;
  constexpr static size_t SRS_SHIFT_3_OFFSET = 14;
  constexpr static size_t SRS_SHIFT_4_OFFSET = 15;
  constexpr static size_t ELT_ADD_PARAMS_OFFSET = 16;
  constexpr static size_t OFM_DEPTH_ITERS_OFFSET = 17;
  constexpr static size_t IFM_DEPTH_ITERS_OFFSET = 18;
  constexpr static size_t OP_TYPE_OFFSET = 19;
  constexpr static size_t OFM_LEN_1_OFFSET = 20;
  constexpr static size_t OFM_LEN_2_OFFSET = 21;
  constexpr static size_t IFM_Y_ITER_OFFSET = 22;
  constexpr static size_t IFM_X_ITER_OFFSET = 23;
  constexpr static size_t CH_IN_DEPTH_SPLIT_OFFSET = 24;
  constexpr static size_t ADF_COLUMS_OFFSET = 25;
  constexpr static size_t CONV2D_OFM_PAD_OFFSET = 26;
  constexpr static size_t ACT_OFFSET = 27;
  constexpr static size_t CONV_MODE_OFFSET = 28;
  constexpr static size_t ROW_OFFSET_OFFSET = 29;
  constexpr static size_t IFM_SIGN_ELT_ADD_PARAMS_OFFSET = 30;

  std::vector<uint8_t> ddFlowParams;

  void set_8_bits(size_t idx, uint8_t val) { ddFlowParams[idx] = val; }

  void set_16_bits(size_t idx, uint16_t val) {
    ddFlowParams[idx] = val & 0xFF;
    ddFlowParams[idx + 1] = (val >> 8) & 0xFF;
  }

  void set_32_bits(size_t idx, uint32_t val) {
    ddFlowParams[idx] = val & 0xFF;
    ddFlowParams[idx + 2] = (val >> 16) & 0xFF;
    ddFlowParams[idx + 1] = (val >> 8) & 0xFF;
    ddFlowParams[idx + 3] = (val >> 24) & 0xFF;
  }

public:
  LayerParams() { ddFlowParams.resize(64, 0); }

  void set_sv_iw(uint8_t val) { set_8_bits(SV_IW_OFFSET, val); }
  void set_sv_ih(uint8_t val) { set_8_bits(SV_IH_OFFSET, val); }
  void set_icg(uint8_t val) { set_8_bits(ICG_OFFSET, val); }
  void set_ocg(uint8_t val) { set_8_bits(OCG_OFFSET, val); }
  void set_k_y(uint8_t val) { set_8_bits(K_Y_OFFSET, val); }
  void set_log2stride(uint8_t val) { set_8_bits(LOG2STRIDE_OFFSET, val); }
  void set_let_add_params(uint8_t val) {
    set_8_bits(LET_ADD_PARAMS_OFFSET, val);
  }
  void set_out_mode(uint8_t val) { set_8_bits(OUT_MODE_OFFSET, val); }
  void set_srs_shift_5(uint8_t val) { set_8_bits(SRS_SHIFT_5_OFFSET, val); }
  void set_srs_shift_6(uint8_t val) { set_8_bits(SRS_SHIFT_6_OFFSET, val); }
  void set_srs_shift_0(uint8_t val) { set_8_bits(SRS_SHIFT_0_OFFSET, val); }
  void set_p_sum_buf_offset(uint8_t val) {
    set_8_bits(P_SUM_BUF_OFFSET_OFFSET, val);
  }
  void set_lrelu_alpha_scaled_1(uint8_t val) {
    set_8_bits(LRELU_ALPHA_SCALED_1_OFFSET, val);
  }
  void set_lrelu_alpha_scaled_2(uint8_t val) {
    set_8_bits(LRELU_ALPHA_SCALED_2_OFFSET, val);
  }
  void set_srs_shift_3(uint8_t val) { set_8_bits(SRS_SHIFT_3_OFFSET, val); }
  void set_srs_shift_4(uint8_t val) { set_8_bits(SRS_SHIFT_4_OFFSET, val); }
  void set_elt_add_params(uint8_t val) {
    set_8_bits(ELT_ADD_PARAMS_OFFSET, val);
  }
  void set_ofm_depth_iters(uint8_t val) {
    set_8_bits(OFM_DEPTH_ITERS_OFFSET, val);
  }
  void set_ifm_depth_iters(uint8_t val) {
    set_8_bits(IFM_DEPTH_ITERS_OFFSET, val);
  }
  void set_op_type(uint8_t val) { set_8_bits(OP_TYPE_OFFSET, val); }
  void set_ofm_len_1(uint8_t val) { set_8_bits(OFM_LEN_1_OFFSET, val); }
  void set_ofm_len_2(uint8_t val) { set_8_bits(OFM_LEN_2_OFFSET, val); }
  void set_ifm_y_iter(uint8_t val) { set_8_bits(IFM_Y_ITER_OFFSET, val); }
  void set_ifm_x_iter(uint8_t val) { set_8_bits(IFM_X_ITER_OFFSET, val); }
  void set_ch_in_depth_split(uint8_t val) {
    set_8_bits(CH_IN_DEPTH_SPLIT_OFFSET, val);
  }
  void set_adf_colums(uint8_t val) { set_8_bits(ADF_COLUMS_OFFSET, val); }
  void set_conv2d_ofm_pad(uint8_t val) {
    set_8_bits(CONV2D_OFM_PAD_OFFSET, val);
  }
  void set_act(uint8_t val) { set_8_bits(ACT_OFFSET, val); }
  void set_conv_mode(uint8_t val) { set_8_bits(CONV_MODE_OFFSET, val); }
  void set_row_offset(uint8_t val) { set_8_bits(ROW_OFFSET_OFFSET, val); }
  void set_ifm_sign_elt_add_param(uint8_t val) {
    set_8_bits(IFM_SIGN_ELT_ADD_PARAMS_OFFSET, val);
  }

  std::vector<uint8_t> data() { return ddFlowParams; }
};

class ConvLayerParams {
  constexpr static size_t N_BYTES_LP = 64;

  constexpr static size_t W_ITER_OFFSET = 0;
  constexpr static size_t H_ITER_OFFSET = 1;
  constexpr static size_t DEPTH_ITER_OFFSET = 2;
  constexpr static size_t CH_ITER_OFFSET = 3;
  constexpr static size_t KX_G_OFFSET = 4;
  constexpr static size_t KY_G_OFFSET = 5;
  constexpr static size_t CI_G_OFFSET = 6;
  constexpr static size_t S_G_OFFSET = 7;
  constexpr static size_t N_G_OFFSET = 8;
  constexpr static size_t X_G_OFFSET = 9;
  constexpr static size_t Y_G_OFFSET = 10;
  constexpr static size_t CO_G_OFFSET = 11;

  constexpr static size_t INNER_G_OFFSET = 12;
  constexpr static size_t OUTER_G_OFFSET = 14;

  constexpr static size_t SHIFT_TDM_OFFSET = 16;
  constexpr static size_t SHIFT_RES_OFFSET = 17;
  constexpr static size_t SHIFT_NORM_OFFSET = 18;
  constexpr static size_t SHIFT_BIAS_OFFSET = 19;
  constexpr static size_t STEP_KX_OFFSET = 20;
  constexpr static size_t STEP_KY_OFFSET = 22;
  constexpr static size_t STEP_CI_OFFSET = 24;
  constexpr static size_t STEP_XI_OFFSET = 26;
  constexpr static size_t STEP_YI_OFFSET = 28;
  constexpr static size_t STEP_XO_OFFSET = 30;
  constexpr static size_t STEP_YO_OFFSET = 32;

  constexpr static size_t STEP_CO_OFFSET = 34;
  constexpr static size_t WRAPPER_TYPE_OFFSET = 36;
  constexpr static size_t MLKERNELCONTROL_OFFSET = 40;
  constexpr static size_t C1_OFFSET = 44;
  constexpr static size_t C2_OFFSET = 48;
  constexpr static size_t SHIFT_QB_OFFSET = 52;
  constexpr static size_t SHIFT_OUT_OFFSET = 53;
  constexpr static size_t IFM_SV_WIDTH_OFFSET = 54;
  constexpr static size_t IFM_SV_HEIGHT_OFFSET = 55;
  constexpr static size_t IFM_SV_DEPTH_OFFSET = 56;
  constexpr static size_t OFM_SV_DEPTH_OFFSET = 57;
  constexpr static size_t MAXPOOL_KSIZE_OFFSET = 58;
  constexpr static size_t MAXPOOL_STRIDE_OFFSET = 59;
  constexpr static size_t MANUAL_BD_PAD_VAL_OFFSET = 60;

  std::vector<uint8_t> layerParams;

  void set_8_bits(size_t idx, uint8_t val) { layerParams[idx] = val; }

  void set_16_bits(size_t idx, uint16_t val) {
    layerParams[idx] = val & 0xFF;
    layerParams[idx + 1] = (val >> 8) & 0xFF;
  }

  void set_32_bits(size_t idx, uint32_t val) {
    layerParams[idx] = val & 0xFF;
    layerParams[idx + 1] = (val >> 8) & 0xFF;
    layerParams[idx + 2] = (val >> 16) & 0xFF;
    layerParams[idx + 3] = (val >> 24) & 0xFF;
  }

public:
  ConvLayerParams() { layerParams.resize(N_BYTES_LP, 0); }

  // 8 bits
  void set_w_iter(uint8_t val) { set_8_bits(W_ITER_OFFSET, val); }
  void set_h_iter(uint8_t val) { set_8_bits(H_ITER_OFFSET, val); }
  void set_depth_iter(uint8_t val) { set_8_bits(DEPTH_ITER_OFFSET, val); }
  void set_ch_iter(uint8_t val) { set_8_bits(CH_ITER_OFFSET, val); }
  void set_kx_g(uint8_t val) { set_8_bits(KX_G_OFFSET, val); }
  void set_ky_g(uint8_t val) { set_8_bits(KY_G_OFFSET, val); }
  void set_ci_g(uint8_t val) { set_8_bits(CI_G_OFFSET, val); }
  void set_s_g(uint8_t val) { set_8_bits(S_G_OFFSET, val); }
  void set_n_g(uint8_t val) { set_8_bits(N_G_OFFSET, val); }
  void set_x_g(uint8_t val) { set_8_bits(X_G_OFFSET, val); }
  void set_y_g(uint8_t val) { set_8_bits(Y_G_OFFSET, val); }
  void set_co_g(uint8_t val) { set_8_bits(CO_G_OFFSET, val); }

  // 16 bits
  void set_inner_g(uint16_t val) { set_16_bits(INNER_G_OFFSET, val); }
  void set_outer_g(uint16_t val) { set_16_bits(OUTER_G_OFFSET, val); }

  // 8 bits
  void set_shift_tdm(uint8_t val) { set_8_bits(SHIFT_TDM_OFFSET, val); }
  void set_shift_res(uint8_t val) { set_8_bits(SHIFT_RES_OFFSET, val); }
  void set_shift_norm(uint8_t val) { set_8_bits(SHIFT_NORM_OFFSET, val); }

  // 16 bits
  void set_shift_bias(uint16_t val) { set_16_bits(SHIFT_BIAS_OFFSET, val); }
  void set_step_kx(uint16_t val) { set_16_bits(STEP_KX_OFFSET, val); }
  void set_step_ky(uint16_t val) { set_16_bits(STEP_KY_OFFSET, val); }
  void set_step_ci(uint16_t val) { set_16_bits(STEP_CI_OFFSET, val); }
  void set_step_xi(uint16_t val) { set_16_bits(STEP_XI_OFFSET, val); }
  void set_step_yi(uint16_t val) { set_16_bits(STEP_YI_OFFSET, val); }
  void set_step_xo(uint16_t val) { set_16_bits(STEP_XO_OFFSET, val); }
  void set_step_yo(uint16_t val) { set_16_bits(STEP_YO_OFFSET, val); }
  void set_step_co(uint16_t val) { set_16_bits(STEP_CO_OFFSET, val); }

  // 32 bits
  void set_wrapper_type(uint32_t val) { set_32_bits(WRAPPER_TYPE_OFFSET, val); }
  void set_mlkernel_control(uint32_t val) {
    set_32_bits(MLKERNELCONTROL_OFFSET, val);
  }
  void set_c1(uint32_t val) { set_32_bits(C1_OFFSET, val); }
  void set_c2(uint32_t val) { set_32_bits(C2_OFFSET, val); }

  // 8 bits
  void set_shift_qb(uint8_t val) { set_8_bits(SHIFT_QB_OFFSET, val); }
  void set_shift_out(uint8_t val) { set_8_bits(SHIFT_OUT_OFFSET, val); }
  void set_ifm_sv_width(uint8_t val) { set_8_bits(IFM_SV_WIDTH_OFFSET, val); }
  void set_ifm_sv_height(uint8_t val) { set_8_bits(IFM_SV_HEIGHT_OFFSET, val); }
  void set_ifm_sv_depth(uint8_t val) { set_8_bits(IFM_SV_DEPTH_OFFSET, val); }
  void set_ofm_sv_depth(uint8_t val) { set_8_bits(OFM_SV_DEPTH_OFFSET, val); }
  void set_maxpool_ksize(uint8_t val) { set_8_bits(MAXPOOL_KSIZE_OFFSET, val); }
  void set_maxpool_stride(uint8_t val) {
    set_8_bits(MAXPOOL_STRIDE_OFFSET, val);
  }
  void set_manual_bd_pad_val(uint8_t val) {
    set_8_bits(MANUAL_BD_PAD_VAL_OFFSET, val);
  }

  std::vector<uint8_t> data() { return this->layerParams; }
};

std::vector<int8_t> get_type_size(const std::vector<std::string> &data_types);

std::vector<std::vector<uint8_t>> computeUtil(bool do_maxpool,
                                              bool do_transpose,
                                              const TilingInfo &tilingInfo,
                                              const LayerInfo &layerInfo);

std::vector<std::vector<uint8_t>> computeLayerParams(const TilingInfo &,
                                                     const LayerInfo &);

void compare_vectors_and_print(const std::vector<uint8_t> &vec1,
                               const std::vector<uint8_t> &vec2);

std::string
generate_tiling_key(const std::vector<int32_t> &input_shape,
                    const std::vector<int32_t> &output_shape,
                    const std::vector<int32_t> &weight_shape,
                    const std::vector<int32_t> &maxpool_kernel_shape,
                    const std::vector<int32_t> &maxpool_stride);
} // namespace conv_lp
#endif // LP_COMPUTATION_H
