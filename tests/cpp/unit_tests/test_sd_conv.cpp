/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#include <cfenv>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <torch/torch.h>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <ops/sd/conv2d.hpp>

// #pragma STDC FENV_ACCESS ON
inline float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

static std::vector<uint32_t> read_hex_file(const std::string &filePath) {
  std::ifstream fileStream(filePath);

  if (!fileStream.is_open()) {
    std::cerr << "Failed to open file " << filePath << "!" << std::endl;
    throw std::runtime_error("Failed to open file " + filePath + "!");
    return {};
  }

  std::cout << "Opened file " << filePath << " for reading hex data!"
            << std::endl;

  std::vector<uint32_t> buffer;
  uint32_t temp;

  while (fileStream >> std::hex >> temp) {
    buffer.push_back(temp);
  }

  fileStream.close();
  return buffer;
}

template <typename T>
int sd_conv_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
                         float error_tolerance = 0.01,
                         float pixel_L2_norm_tolerance = 0.01) {
  int fail = 0;
  float max_diff = 0;
  float L2_norm = 0;
  int err_count = 0;
  for (int i = 0; i < cpu_Y.size(); ++i) {
    float diff = std::abs(bfloat16_to_float(cpu_Y.at(i)) -
                          bfloat16_to_float(aie_Y.at(i)));
    L2_norm += ((float)diff * (float)diff);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > error_tolerance) {
      // if (err_count < 100) {
      //   std::cout << "ERROR: Y[" << i << "]: "
      //             << "Expected: " << bfloat16_to_float(cpu_Y.at(i)) << ","
      //             << "Received: " << bfloat16_to_float(aie_Y.at(i))
      //             << "\n";
      // }
      fail = 1;
      err_count++;
    }
  }
  L2_norm = std::sqrt(L2_norm);
  auto pixel_L2_norm = L2_norm / cpu_Y.size();
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  std::cout << "pixel L2_norm is " << pixel_L2_norm << std::endl;
  std::cout << "pixel_L2_norm_tolerance is " << pixel_L2_norm_tolerance
            << std::endl;
  if (err_count > 0 && pixel_L2_norm < pixel_L2_norm_tolerance) {
    std::cout << "deem err_count as zero due to low pixel_L2_norm" << std::endl;
    err_count = 0;
  }
  std::cout << "err_count is " << err_count << std::endl;
  return err_count;
}

inline double round_half_to_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

static void aie_srs(std::vector<uint32_t> &input_output) {
  int data_width = 16;
  int shift = 16;
  for (size_t i = 0; i < input_output.size(); ++i) {
    double temp = static_cast<double>(input_output[i]) / std::pow(2.0, shift);
    // temp = std::round(temp);
    temp = round_half_to_even(temp);
    if (temp > std::pow(2.0f, data_width) - 1) {
      temp = float(std::pow(2.0f, data_width)) - 1;
    }
    if (temp < 0) {
      temp = 0;
    }
    input_output[i] = static_cast<uint32_t>(temp);
  }
}

static void float2bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs(x_uint32);
  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint32[i] = (static_cast<uint16_t>(x_uint32[i]) << 16);
  }
  std::memcpy(x.data(), x_uint32.data(), x.size() * sizeof(float));
}

// Helper function to reinterpret float as uint32_t
static uint32_t float_as_uint(float v) {
  union {
    float f;
    uint32_t i;
  } u;
  u.f = v;
  return u.i;
}

// Get the exponent of the floating-point number in IEEE 754 format
inline int get_exponent_cpu(float v) {
  uint32_t uint_v = float_as_uint(v);
  return (uint_v & 0x7f800000) >> 23;
}

// Python-like rounding function
static float py3_round(float x) {
  float x_floor = std::floor(x);
  float diff = x - x_floor;

  if (diff > 0.5) {
    return x_floor + 1;
  } else if (diff == 0.5) {
    return (static_cast<int>(x_floor) % 2 == 0) ? x_floor : (x_floor + 1);
  } else {
    return x_floor;
  }
}

// Offline quantization, weight quantization
std::vector<float> bfp_cpu_kernel(const std::vector<float> &input, int n,
                                  int index, int stride, int bit_width) {
  int shared_exp = 0;
  std::vector<float> output(input.size(), 0.0f);

  // Loop over block to find shared exponent
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  // Minus 127 to get unbiased value
  int shared_exp_value = shared_exp - 127;
  // 1 sign bit, 8 exp bits
  int m_bits = bit_width - 9;
  float scale = std::pow(2.0f, shared_exp_value - (m_bits - 1));
  float max_v = std::pow(2.0f, shared_exp_value + 1) - scale;

  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      float x = py3_round(input[i] / scale) * scale;
      output[i] = std::max(-max_v, std::min(x, max_v));
    }
  }

  return output;
}

std::vector<float> bfp_cpu_kernel_hw(const std::vector<float> &input, int n,
                                     int index, int stride, int bit_width) {
  int shared_exp = 0;
  std::vector<float> output(n, 0.0f);

  // First pass to determine the shared exponent
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  int shared_exp_value = shared_exp - 127;
  int m_bits = bit_width - 9;
  float scale = std::pow(2.0f, shared_exp_value - (m_bits - 1));

  // Adjust shared exponent if needed
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == shared_exp) {
      float x = py3_round(input[i] / scale);
      if (x >= 128 || x < -128) {
        shared_exp++;
        shared_exp_value++;
        scale *= 2.0f;
        break;
      }
    }
  }

  float max_v = std::pow(2.0f, shared_exp_value) * (std::pow(2.0f, m_bits) - 1);
  float min_v = -std::pow(2.0f, shared_exp_value) * std::pow(2.0f, m_bits);

  // Final pass to quantize values
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      float x = py3_round(input[i] / scale) * scale;
      output[i] = std::max(min_v, std::min(x, max_v));
    }
  }
  return output;
}

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

template <typename OuT>
void torch_conv2d(std::vector<float> ifm, std::vector<float> wts,
                  std::vector<float> bias, std::vector<OuT> &ofm, int IC,
                  int IH, int IW, int OC, int OH, int OW, int kh, int kw,
                  int strideH, int strideW, int padding, int batch) {
  EXPECT_TRUE(ifm.size() == batch * IC * IH * IW)
      << " wrong ifm size " << ifm.size() << " , tensor size is "
      << batch * IC * IH * IW;
  EXPECT_TRUE(wts.size() == OC * IC * kh * kh)
      << " wrong wts size " << wts.size() << " , tensor size is "
      << OC * IC * kh * kw;
  EXPECT_TRUE(bias.size() == OC)
      << " wrong bias size " << ifm.size() << " , tensor size is " << OC;
  EXPECT_TRUE(ofm.size() == batch * OC * OH * OW)
      << " wrong ifm size " << ifm.size() << " , tensor size is "
      << batch * OC * OH * OW;
  auto torch_input_tensor =
      torch::from_blob(ifm.data(), {batch, IC, IH, IW}, torch::kFloat);
  torch::nn::Conv2d conv_layer(
      torch::nn::Conv2dOptions(IC, OC, kh).stride(strideH).padding(padding));
  // wts nchw
  auto torch_weight_tensor =
      torch::from_blob(wts.data(), {OC, IC, kh, kw}, torch::kFloat);
  auto torch_bias_tensor = torch::from_blob(bias.data(), {OC}, torch::kFloat);
  conv_layer->weight = torch_weight_tensor.clone();
  conv_layer->bias = torch_bias_tensor.clone();
  auto torch_output_tensor = conv_layer->forward(torch_input_tensor);
  std::vector<float> torch_output_buffer(torch_output_tensor.numel());
  std::memcpy(torch_output_buffer.data(), torch_output_tensor.data_ptr<float>(),
              torch_output_tensor.numel() * sizeof(float));
  float2bf16_vec(torch_output_buffer);
  uint32_t *torch_output_buffer_as_u =
      reinterpret_cast<uint32_t *>(torch_output_buffer.data());
  // truncate bf16 in float to bf16
  std::vector<OuT> torch_output_bf16(batch * OC * OH * OW);
  for (int i = 0; i < torch_output_buffer.size(); i++) {
    torch_output_bf16[i] = torch_output_buffer_as_u[i] >> 16;
  }
  // transpose ofm to nhwc
  for (int n = 0; n < batch; n++) {
    for (int oc = 0; oc < OC; oc++) {
      for (int h = 0; h < OH; h++) {
        for (int w = 0; w < OW; w++) {
          ofm[n * OH * OW * OC + h * OW * OC + w * OC + oc] =
              torch_output_bf16[n * OC * OH * OW + oc * OH * OW + h * OW + w];
        }
      }
    }
  }
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_sd_conv(int IC, int IH, int IW, int OC, int OH, int OW, int kh, int kw,
                 int strideH, int strideW, int padding, bool debug = false,
                 const std::string &ifm_type = "bfloat16", // a bo
                 const std::string &wgt_type = "bfloat16", // b bo
                 const std::string &out_type = "bfloat16", // c bo
                 const std::string &model_name = "SD_VAE_DEC", size_t batch = 1,
                 float pixel_L2_norm_tolerance = 0.01,
                 bool test_with_golden = false) {
  int quantize_err_count = 0;
  int unquantize_err_count = 0;
  float error_tolerance = 0.01;
  std::map<std::string, std::string> txnbin_a_header = {
      {"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  std::map<std::string, std::string> txnbin_b_header = {
      {"float", "w16bf"}, {"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  size_t ICs = static_cast<size_t>(IC);
  size_t IHs = static_cast<size_t>(IH);
  size_t IWs = static_cast<size_t>(IW);
  size_t OCs = static_cast<size_t>(OC);
  size_t OHs = static_cast<size_t>(OH);
  size_t OWs = static_cast<size_t>(OW);
  size_t khs = static_cast<size_t>(kh);
  size_t kws = static_cast<size_t>(kw);
  size_t OCP = static_cast<size_t>(OC);
  // Note (xcl): Currently only aligned when the size is less than 4.
  // Consider extending this to align to 4-byte boundaries in the future.
  if (OC < 4) {
    OCP = 4;
  }

  std::vector<size_t> a_shape = {batch, IHs, IWs, ICs};
  std::vector<size_t> aie_out_shape = {batch, OHs, OWs, OCP};
  std::vector<OuT> aie_out(batch * OCP * OH * OW);

  int wgt_size = 0;

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{int(batch), IH, IW, IC};
  attr["output_shape"] = std::vector<int>{int(batch), OH, OW, OC};
  attr["weight_shape"] = std::vector<int>{OC, kh, kw, IC};
  // std::cout << "OC = " << OC << ", OH = " << OH << ", OW = " << OW << ", IC =
  // " << IC << ", IH = " << IH << ", IW = " << IW << std::endl;
  if (test_with_golden) {
    const std::string bias_type = "float32";
    ryzenai::sd::conv sd_conv =
        ryzenai::sd::conv<std::uint16_t, std::uint8_t, float, std::uint16_t>(
            ifm_type, wgt_type, bias_type, out_type, false, attr);
    sd_conv.debug(debug);
    ryzenai::sd_conv2d_shapes shapes(OCs, ICs, IHs, IWs, OHs, OWs, khs, kws);
    sd_conv.set_params(model_name, shapes);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_conv/";
    std::string shape_key =
        txnbin_a_header.at(ifm_type) + txnbin_b_header.at(wgt_type) +
        txnbin_acc_header.at(out_type) + "_" + std::to_string(OC) + "_" +
        std::to_string(IC) + "_" + std::to_string(IH) + "_" +
        std::to_string(IW) + "_" + std::to_string(OH) + "_" +
        std::to_string(OW) + "_" + std::to_string(kh) + "_" +
        std::to_string(kw);
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> b_aie = read_hex_file(wts_path);

    std::vector<size_t> b_shape = {b_aie.size() * sizeof(uint32_t)};
    std::vector<Tensor> const_Tensor;
    // bias is actually not used because it is merged to b_aie.
    const_Tensor = {{b_aie.data(), b_shape, wgt_type}};
    sd_conv.initialize_const_params(const_Tensor);
    std::vector<Tensor> input_Tensor;

    input_Tensor = {{a_aie.data(), a_shape, ifm_type}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("OC = " << OC << ", OH = " << OH << ", OW = " << OW
                     << ", IC = " << IC << ", IH = " << IH << ", IW = " << IW);
    PROFILE_THIS(sd_conv.execute(input_Tensor, output_Tensor));
#else
    sd_conv.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    quantize_err_count = sd_conv_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    const std::string bias_type = "float32";
    ryzenai::sd::conv sd_conv =
        ryzenai::sd::conv<std::uint16_t, float, float, std::uint16_t>(
            "bfloat16", "float32", "float32", "bfloat16", false, attr);
    sd_conv.debug(debug);
    ryzenai::sd_conv2d_shapes shapes(OCs, ICs, IHs, IWs, OHs, OWs, khs, kws);
    sd_conv.set_params(model_name, shapes);
    // gen rand
    std::vector<float> raw_bias(OC, 0);
    initialize_random_float(raw_bias, 2, -2);
    auto bf16_bias = raw_bias;
    float2bf16_vec(bf16_bias);

    std::vector<float> raw_wts(OC * kh * kw * IC, 0);
    initialize_random_float(raw_wts, 2, -2);
    auto bf16_wts = raw_wts;
    float2bf16_vec(bf16_wts);

    std::vector<float> raw_ifms(batch * IH * IW * IC, 0);
    initialize_random_float(raw_ifms, 2, -2);
    auto bf16_ifms = raw_ifms;
    float2bf16_vec(bf16_ifms);

    // wts from bfloat16 to bfp16ebs8
    for (int bfp_oc = 0; bfp_oc < OC; bfp_oc++) {
      for (int bfp_h = 0; bfp_h < kh; bfp_h++) {
        for (int bfp_w = 0; bfp_w < kw; bfp_w++) {
          for (int bfp_ic = 0; bfp_ic < IC; bfp_ic += 8) {
            // process bfp_ic to bfp_ic+8
            std::vector<float> wts_aie_bfp(8);
            for (int i = 0; i < 8; i++) {
              if (bfp_ic + i >= IC) {
                wts_aie_bfp[i] = 1;
              } else {
                wts_aie_bfp[i] =
                    bf16_wts[bfp_oc * kh * kw * IC + bfp_h * kw * IC +
                             bfp_w * IC + bfp_ic + i];
              }
            }
            std::vector<float> wts_aie_bfp_quantized =
                bfp_cpu_kernel(wts_aie_bfp, 8, 0, 1, 16);
            // assign back
            for (int i = 0; i < 8; i++) {
              if (bfp_ic + i >= IC) {
                break;
              }
              bf16_wts[bfp_oc * kh * kw * IC + bfp_h * kw * IC + bfp_w * IC +
                       bfp_ic + i] = wts_aie_bfp_quantized[i];
            }
          }
        }
      }
    }

    // shuffle wts32_bfp and bias32_bf16
    std::vector<Tensor> const_tensors;
    std::vector<size_t> in_shape = {(size_t)1, (size_t)IH, (size_t)IW,
                                    (size_t)IC};
    std::vector<size_t> weight_shape = {(size_t)OC, (size_t)kh, (size_t)kw,
                                        (size_t)IC};

    // float wts input expected to be in {oc, kh, kw, ic} order
    // need to transpose, now we are in{oc, ic, kh, kw} order
    std::vector<float> wts_oc_hw_ic(OC * kh * kw * IC);
    for (int oc = 0; oc < OC; ++oc) {
      for (int i_c = 0; i_c < IC; ++i_c) {
        for (int h = 0; h < kh; ++h) {
          for (int w = 0; w < kw; ++w) {
            int oc_ic_hw_idx = oc * IC * kh * kw + i_c * kh * kw + h * kw + w;
            int oc_hw_ic_idx = oc * kh * kw * IC + h * kw * IC + w * IC + i_c;
            wts_oc_hw_ic[oc_hw_ic_idx] = bf16_wts[oc_ic_hw_idx];
          }
        }
      }
    }

    const_tensors.push_back({wts_oc_hw_ic.data(), weight_shape, "float"});
    std::vector<size_t> bias_shape = {(size_t)OC};
    const_tensors.push_back({bf16_bias.data(), bias_shape, "float"});

    size_t total_bytes = 0;
    std::vector<uint16_t> act_p(IH * IW * IC);
    std::vector<Tensor> my_input = {{act_p.data(), in_shape, "bfloat16"}};
    std::vector<Tensor> my_output;
    std::vector<OpArgMap> arg_map =
        sd_conv.get_buffer_reqs(my_input, my_output, attr);
    for (auto arg : arg_map) {
      if (arg.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
        total_bytes = arg.size;
      }
    }

    sd_conv.initialize_const_params(const_tensors);

    auto bf16_ifms_backup = bf16_ifms;

    // ifm is in nchw format
    for (int n = 0; n < batch; n++) {
      for (int bfp_ih = 0; bfp_ih < IH; bfp_ih++) {
        for (int bfp_iw = 0; bfp_iw < IW; bfp_iw++) {
          for (int bfp_ic = 0; bfp_ic < IC; bfp_ic += 8) {
            std::vector<float> ifm_aie_bfp(8);
            bool save_temp = false;
            for (int i = 0; i < 8; i++) {
              if (bfp_ic + i >= IC) {
                ifm_aie_bfp[i] = 1;
              } else {
                ifm_aie_bfp[i] =
                    bf16_ifms[n * IC * IH * IW + (bfp_ic + i) * IH * IW +
                              bfp_ih * IW + bfp_iw];
              }
            }
            std::vector<float> ifm_aie_bfp_quantized =
                bfp_cpu_kernel_hw(ifm_aie_bfp, 8, 0, 1, 16);
            for (int i = 0; i < 8; i++) {
              if (bfp_ic + i >= IC) {
                break;
              }
              bf16_ifms[n * IC * IH * IW + (bfp_ic + i) * IH * IW +
                        bfp_ih * IW + bfp_iw] = ifm_aie_bfp_quantized[i];
            }
          }
        }
      }
    }

    // convert ifm from float to bf16
    uint32_t *cpp_ifm32_bf16_as_u =
        reinterpret_cast<uint32_t *>(bf16_ifms_backup.data());
    std::vector<uint16_t> aie_ifm_bf16(batch * IC * IH * IW);
    EXPECT_EQ(bf16_ifms.size(), aie_ifm_bf16.size());
    // transpose ifm from nchw to nhwc, and truncate to bf16
    for (int n = 0; n < batch; n++) {
      for (int ic = 0; ic < IC; ic++) {
        for (int h = 0; h < IH; h++) {
          for (int w = 0; w < IW; w++) {
            aie_ifm_bf16[n * IC * IH * IW + h * IW * IC + w * IC + ic] =
                cpp_ifm32_bf16_as_u[n * IC * IH * IW + ic * IH * IW + h * IW +
                                    w] >>
                16;
          }
        }
      }
    }

    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_conv/";
    std::string shape_key =
        txnbin_a_header.at(ifm_type) + txnbin_b_header.at(wgt_type) +
        txnbin_acc_header.at(out_type) + "_" + std::to_string(OC) + "_" +
        std::to_string(IC) + "_" + std::to_string(IH) + "_" +
        std::to_string(IW) + "_" + std::to_string(OH) + "_" +
        std::to_string(OW) + "_" + std::to_string(kh) + "_" +
        std::to_string(kw);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{aie_ifm_bf16.data(), a_shape, ifm_type}};
    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("OC = " << OC << ", OH = " << OH << ", OW = " << OW
                     << ", IC = " << IC << ", IH = " << IH << ", IW = " << IW);
    PROFILE_THIS(sd_conv.execute(input_Tensor, output_Tensor));
#else
    sd_conv.execute(input_Tensor, output_Tensor);
#endif
    auto rst_aie_out = aie_out;
    if (OC == 3) {
      std::vector<OuT> tmp_aie_out(batch * OC * OH * OW);
      uint32_t src_stride = 4;
      for (int i = 0; i < batch * OH * OW; i++) {
        memcpy(
            (void *)((uint8_t *)tmp_aie_out.data() + i * OC * sizeof(OuT)),
            (void *)((uint8_t *)aie_out.data() + i * src_stride * sizeof(OuT)),
            (OC * sizeof(OuT)));
      }
      rst_aie_out.resize(batch * OC * OH * OW);
      memcpy(rst_aie_out.data(), tmp_aie_out.data(),
             tmp_aie_out.size() * sizeof(OuT));
    }

    std::vector<OuT> bf16_ofm(batch * OC * OH * OW);
    torch_conv2d<OuT>(bf16_ifms, bf16_wts, bf16_bias, bf16_ofm, IC, IH, IW, OC,
                      OH, OW, kh, kw, strideH, strideW, padding, batch);
    std::cout << "diff with quantized ifm torch conv2d" << std::endl;
    quantize_err_count = sd_conv_check_result<OuT>(
        bf16_ofm, rst_aie_out, error_tolerance, pixel_L2_norm_tolerance);
    std::cout << std::endl;
    std::cout << "diff with unquantized ifm torch conv2d" << std::endl;
    std::vector<OuT> fp32_ofm(batch * OC * OH * OW);
    torch_conv2d<OuT>(raw_ifms, raw_wts, raw_bias, fp32_ofm, IC, IH, IW, OC, OH,
                      OW, kh, kw, strideH, strideW, padding, batch);
    unquantize_err_count = sd_conv_check_result<OuT>(
        fp32_ofm, rst_aie_out, error_tolerance, pixel_L2_norm_tolerance);
    std::cout << "out unquantize_err_count " << unquantize_err_count
              << std::endl;
  }
  return quantize_err_count;
}

// Golden unittest start
TEST(SD_CONV_Test, Golden_Kernel1) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 8, 128, 512, 8, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer10) {
  // layer10_10.87ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 256, 256, 256, 256, 256, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer4) {
  // layer4_23.08ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 512, 128, 512, 512, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer2) {
  // layer2_1.87ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  // this case keep high pixel_L2_norm_tolerance for unknown reason
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 512, 512, 3, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.11, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer5) {
  // layer5_26.94ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 512, 128, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer3) {
  // layer3_11.31ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 256, 256, 256, 256, 256, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer6) {
  // layer6_44.99ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 512, 256, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer7) {
  // layer7_0.047ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 4, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer8) {
  // layer8_0.64ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 512, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer9) {
  // layer9
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 128, 128, 512, 128, 128, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer11) {
  // layer11
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 256, 256, 256, 256, 256, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer12) {
  // layer12
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 256, 256, 512, 256, 256, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelVaelayer13) {
  // layer13
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  // this case keep high pixel_L2_norm_tolerance for unknown reason
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 64, 64, 512, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.13, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer1) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer2) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer4) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 1280, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.3, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer5) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer6) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer7) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 8, 8, 1280, 8, 8, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer8) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 8, 8, 1280, 8, 8, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer9) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer10) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer11) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer13) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer14) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer15) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer16) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 8, 8, 1280, 8, 8, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer17) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer18) {
  // layer18
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.2, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer19) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 8, 8, 1280, 8, 8, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer20) {
  // layer20
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer24) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer25) {
  // layer25
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.3, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer26) {
  // layer26
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer27) {
  // layer27
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.3, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer29) {
  // layer29
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 320, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer30) {
  // layer30
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.2, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer31) {
  // layer31
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 640, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer32) {
  // layer32
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer33) {
  // layer33
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer34) {
  // layer34
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 64, 64, 320, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer35) {
  // layer35
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer3) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 8, 8, 3, 3, 2, 2, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer12) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer21) {
  // layer21
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 32, 32, 3, 3, 2, 2, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer22) {
  // layer22
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 4, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer23) {
  // layer23
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Golden_KernelUnetlayer28) {
  // layer28
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 16, 16, 3, 3, 2, 2, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer1) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 128, 128, 1536, 64, 64, 2, 2, 2, 2, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer2) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 1024, 1024, 256, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer3) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 1024, 1024, 128, 1024, 1024, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer4) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 1024, 1024, 3, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer5) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 64, 64, 512, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer6) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 512, 256, 512, 512, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer7) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 512, 512, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer8) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 1024, 1024, 128, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer9) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 64, 64, 1536, 32, 32, 2, 2, 2, 2, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer10) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 128, 128, 512, 128, 128, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer11) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 1024, 1024, 128, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, GoldenSD3NewLayer12) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 512, 256, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// Golden unittest end

// Random unittest start
TEST(SD_CONV_Test, Random_KernelVaelayer1) {
  // layer1_10.71ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 512, 512, 128, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer10) {
  // layer10_10.87ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 256, 256, 256, 256, 256, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer4) {
  // layer4_23.08ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 512, 128, 512, 512, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer2) {
  // layer2_1.87ms
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  // this case keep high pixel_L2_norm_tolerance for unknown reason
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 512, 512, 3, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.11);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer5) {
  // layer5_26.94ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 512, 128, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer3) {
  // layer3_11.31ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 256, 256, 256, 256, 256, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer6) {
  // layer6_44.99ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 512, 256, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer7) {
  // layer7_0.047ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 4, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer8) {
  // layer8_0.64ms, use new ofm ref format
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 512, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer9) {
  // layer9
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 128, 128, 512, 128, 128, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer11) {
  // layer11
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 256, 256, 256, 256, 256, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer12) {
  // layer12
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 256, 256, 512, 256, 256, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelVaelayer13) {
  // layer13
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  // this case keep high pixel_L2_norm_tolerance for unknown reason
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 64, 64, 512, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.13);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer1) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer2) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer4) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 1280, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.3);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer5) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer6) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer7) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 8, 8, 1280, 8, 8, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer8) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 8, 8, 1280, 8, 8, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer9) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer10) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer11) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer13) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer14) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 0.5);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer15) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer16) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 8, 8, 1280, 8, 8, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer17) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer18) {
  // layer18
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.2);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer19) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 8, 8, 1280, 8, 8, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer20) {
  // layer20
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer24) {
  //  IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      2560, 16, 16, 1280, 16, 16, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer25) {
  // layer25
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 16, 16, 1280, 16, 16, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.3);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer26) {
  // layer26
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer27) {
  // layer27
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.3);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer29) {
  // layer29
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 320, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer30) {
  // layer30
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.2);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer31) {
  // layer31
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 640, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer32) {
  // layer32
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 32, 32, 640, 32, 32, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer33) {
  // layer33
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer34) {
  // layer34
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 64, 64, 320, 64, 64, 1, 1, 1, 1, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer35) {
  // layer35
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      960, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer3) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 8, 8, 3, 3, 2, 2, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer12) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      1920, 32, 32, 640, 32, 32, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_UNet", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer21) {
  // layer21
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 32, 32, 3, 3, 2, 2, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer22) {
  // layer22
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 4, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer23) {
  // layer23
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 320, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Random_KernelUnetlayer28) {
  // layer28
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 16, 16, 3, 3, 2, 2, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer1) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 128, 128, 1536, 64, 64, 2, 2, 2, 2, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer2) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 1024, 1024, 256, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer3) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 1024, 1024, 128, 1024, 1024, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer4) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 1024, 1024, 3, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer5) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 64, 64, 512, 64, 64, 3, 3, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer6) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 512, 256, 512, 512, 1, 1, 1, 1, 0, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer7) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 512, 512, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer8) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 1024, 1024, 128, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer9) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 64, 64, 1536, 32, 32, 2, 2, 2, 2, 0, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC", 2, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer10) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      16, 128, 128, 512, 128, 128, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer11) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 1024, 1024, 128, 1024, 1024, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, RandomSD3NewLayer12) {
  // IC, IH, IW, OC, OH, OW, kh, kw, strideH, strideW, padding
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 512, 256, 512, 512, 3, 3, 1, 1, 1, false, "bfloat16",
      "bfp16ebs8", "bfloat16", "SD_VAE_DEC", 1, 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// Random unittest end
