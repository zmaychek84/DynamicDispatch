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
#ifndef _TEST_COMMON_HPP_
#define _TEST_COMMON_HPP_

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "ops/ops_common/dtype_utils.h"

constexpr int32_t garbage_value = 0xCD;
const std::string mdsqr_A8W8_QDQ_XCLBIN_REL_PATH =
    "xclbin/stx/4x2_psf_model_a8w8_qdq.xclbin";
const std::string mxpzi_A16W8_QDQ_XCLBIN_REL_PATH =
    "xclbin/stx/4x2_psj_model_a16w8_qdq.xclbin";
const std::string mxgan_A16W8_QDQ_XCLBIN_REL_PATH =
    "xclbin/stx/4x2_psh_model_a16w8_qdq.xclbin";
const std::string MLADF_GEMM_4x4_A16FW4ACC16F_XCLBIN_PATH =
    "xclbin/stx/mladf_gemm_4x4_a16fw4acc16f.xclbin";
const std::string XCOM_4x4_XCLBIN_REL_PATH = "xclbin/stx/4x4_dpu.xclbin";
const std::string XCOM_4x4_Q_XCLBIN_REL_PATH =
    "xclbin/stx/4x4_dpu_qconv_qelew_add.xclbin";
const std::string MLADF_SOFTMAX_A16_XCLBIN_PATH =
    "xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin";
const std::string
    LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH =
        "xclbin/stx/llama2_mladf_2x4x4_gemmbfp16_silu_mul_mha_rms_rope.xclbin";
const std::string MLADF_2x4x2_GEMM_A16A16_XCLBIN_PATH =
    "xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin";
const std::string MLADF_4x2_GEMM_A16W8_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_gemm_a16w8_qdq.xclbin";

const std::string MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH =
    "xclbin/stx/mladf_4x2_add_a16.xclbin";
const std::string MLADF_ELWMUL_A16W16_QDQ_XCLBIN_PATH =
    "xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin";
template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size,
                              T data_max = std::numeric_limits<T>::max(),
                              T data_min = std::numeric_limits<T>::min()) {
  for (size_t i = 0; i < size; i++) {
    vec.at(i) = (rand() % (data_max - data_min)) + data_min;
  }
}

namespace dd {

static uint16_t rand_bfloat16(float range = 1.0) {
  float x = range * (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f);
  return ryzenai::float_to_bfloat16(x);
}

static void initialize_random_bfloat16(std::vector<uint16_t> &vec,
                                       int data_max) {
  auto data_min = -(data_max + 1);
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = rand_bfloat16(float(data_max));
  }
}

static void initialize_lowertriangular(std::vector<uint16_t> &vec, int M, int N,
                                       uint16_t value) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      if (n <= m) {
        vec[N * m + n] = value;
      }
    }
  }
}

static int count_errors_floatvsbfloat16(std::vector<float> cpu_Y,
                                        std::vector<uint16_t> aie_Y,
                                        std::vector<size_t> tensor_shape,
                                        float error_tolerance = 0.01) {
  size_t num_rows;
  size_t num_cols;
  size_t num_batch;
  if (tensor_shape.size() == 3) {
    num_batch = tensor_shape[0];
    num_rows = tensor_shape[1];
    num_cols = tensor_shape[2];
  } else if (tensor_shape.size() == 2) {
    // no harm with batch size being 1
    num_batch = 1;
    num_rows = tensor_shape[0];
    num_cols = tensor_shape[1];
  } else {
    throw std::runtime_error(
        "count_errors_floatvsbfloat16 only supports either rank 2 [Rows,Cols] "
        "or rank3 [Batch,Rows,Cols] comparisson");
  }
  int fail = 0;
  int err_count = 0;
  float max_diff = 0.0;
  for (int b = 0; b < num_batch; ++b) {
    for (int r = 0; r < num_rows; ++r) {
      for (int c = 0; c < num_cols; ++c) {
        float diff =
            std::abs(cpu_Y.at(b * num_cols * num_rows + r * num_cols + c) -
                     ryzenai::bfloat16_to_float(
                         aie_Y.at(b * num_cols * num_rows + r * num_cols + c)));
        if (diff > max_diff) {
          max_diff = diff;
        }
        if (diff > error_tolerance) {
          // printf("ERROR: Y[%d][%d][%d] Expected: %f, %d, Received: %f,
          // %d\n",b, r, c,
          //        cpu_Y.at(b * num_cols * num_rows + r * num_cols +c),
          //        cpu_Y.at(b * num_cols * num_rows + r * num_cols + c),
          //        bfloat16_to_float(aie_Y.at(b * num_cols * num_rows + r *
          //        num_cols + c)), aie_Y.at(b * num_cols * num_rows + r *
          //        num_cols + c));
          fail = 1;
          err_count++;
        }
      }
    }
  }
  std::cout << "max_diff is " << max_diff << std::endl;
  return err_count;
}

static int count_errors_bfloat16vsbfloat16(std::vector<uint16_t> cpu_Y,
                                           std::vector<uint16_t> aie_Y,
                                           std::vector<size_t> tensor_shape,
                                           float error_tolerance = 0.01) {
  size_t num_rows;
  size_t num_cols;
  size_t num_batch;
  if (tensor_shape.size() == 3) {
    num_batch = tensor_shape[0];
    num_rows = tensor_shape[1];
    num_cols = tensor_shape[2];
  } else if (tensor_shape.size() == 2) {
    // no harm with batch size being 1
    num_batch = 1;
    num_rows = tensor_shape[0];
    num_cols = tensor_shape[1];
  } else {
    throw std::runtime_error(
        "count_errors_floatvsbfloat16 only supports either rank 2 [Rows,Cols] "
        "or rank3 [Batch,Rows,Cols] comparisson");
  }
  int fail = 0;
  int err_count = 0;
  float max_diff = 0.0;
  for (int b = 0; b < num_batch; ++b) {
    for (int r = 0; r < num_rows; ++r) {
      for (int c = 0; c < num_cols; ++c) {
        float diff =
            std::abs(ryzenai::bfloat16_to_float(
                         cpu_Y.at(b * num_cols * num_rows + r * num_cols + c)) -
                     ryzenai::bfloat16_to_float(
                         aie_Y.at(b * num_cols * num_rows + r * num_cols + c)));
        if (diff > max_diff) {
          max_diff = diff;
        }
        if (diff > error_tolerance) {
          // printf("ERROR: Y[%d][%d][%d] Expected: %f, %d, Received: %f, %d
          // \n",
          //        b, r, c, cpu_Y.at(b * num_cols * num_rows + r * num_cols +
          //        c), cpu_Y.at(b * num_cols * num_rows + r * num_cols + c),
          //        ryzenai::bfloat16_to_float(
          //            aie_Y.at(b * num_cols * num_rows + r * num_cols + c)),
          //        aie_Y.at(b * num_cols * num_rows + r * num_cols + c));
          fail = 1;
          err_count++;
        }
      }
    }
  }
  std::cout << "max_diff is " << max_diff << std::endl;
  return err_count;
}

} // namespace dd

static inline void confirmOpen(std::ofstream &file) {
  if (!file) {
    std::cerr << "Error: File could not be opened." << std::endl;
    throw;
  }
}

static void rand_init_int(int8_t *ptr, size_t size, uint32_t seed = 32) {
  srand(seed);
  for (int i = 0; i < size; i++) {
    int8_t r = 16; // static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    ptr[i] = r;
  }
}

template <typename T> static void rrand_init_int(T *ptr, size_t size) {
  static const T max_val = static_cast<T>((0x1 << (8 * sizeof(T))) - 1);
  for (size_t i = 0; i < size; i++) {
    T r = static_cast<T>(rand() & max_val);
    ptr[i] = r;
  }
}

template <typename T> static void init_int_00s(T *ptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    T r = static_cast<T>(0x00);
    ptr[i] = r;
  }
}

template <typename T> static void init_int_01s(T *ptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    T r = static_cast<T>(0x01);
    ptr[i] = r;
  }
}

template <typename T> static void init_int_10s(T *ptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    T r = static_cast<T>(0x10);
    ptr[i] = r;
  }
}

template <typename T> static void init_int_80s(T *ptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    T r = static_cast<T>(0x80);
    ptr[i] = r;
  }
}

static void init_int_ffs(int8_t *ptr, size_t size) {
  for (int i = 0; i < size; i++) {
    int8_t r = -1;
    ptr[i] = r;
  }
}

template <typename T>
static void initialize_random_mladf(std::vector<T> &vec, size_t size,
                                    int data_max, std::string dtype = "int8") {
  auto data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    if (dtype == "bfloat16") {
      vec[i] = ryzenai::rand_bfloat16(float(data_max));
    } else if (dtype == "uint4") {
      vec[i] = ryzenai::rand_uint4(data_max);
    } else if (dtype == "int4") {
      vec[i] = ryzenai::rand_int4(data_max);
    } else if (std::is_same<T, float>::value) {
      vec[i] = (T)(2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f) * data_max;
    } else {
      vec[i] = (T)(rand() % (data_max - data_min + 1)) + data_min;
    }
  }
}

static inline int AllClose(const std::vector<float> &golden,
                           const std::vector<int16_t> &actual, float atol,
                           float rtol) {
  if (golden.size() != actual.size()) {
    throw std::invalid_argument("Vectors must be of the same size");
  }

  int count_not_close = 0;
  for (size_t i = 0; i < golden.size(); ++i) {

    auto golden_adj = ryzenai::bfloat16_rnd_even(golden[i]);
    auto actual_adj = ryzenai::bfloat16_to_float(actual[i]);

    float diff = std::fabs(golden_adj - actual_adj);
    float max_val = std::fmax(std::fabs(golden_adj), std::fabs(actual_adj));
    if (diff > atol + rtol * max_val || std::isnan(actual_adj)) {
      ++count_not_close;
      std::cout << "Index: " << i << " Expected: " << golden_adj
                << " Received: " << actual_adj << std::endl;
    }
  }

  return count_not_close;
}

using SDInfoMap = std::map<std::string, std::string>;

static std::string sd_get_xclbin(const std::string &model_name) {
  static SDInfoMap xclbin_map{
      {"SD15_UNET", "SD15_unet_2x4x4.xclbin"},
      {"SD15_VAE", "SD15_vae_2x4x4.xclbin"},
      {"SD3_DIT1024", "SD3_MMDIT_2x4x4.xclbin"},
      {"SD3_VAE1024", "SD3_1K_VAE_2x4x4.xclbin"},
      {"SD3_DIT512", "SD3_MMDIT_2x4x4.xclbin"}, // same as 1k
      {"SD3_VAE512", "SD3_VAE_2x4x4.xclbin"},
  };
  if (xclbin_map.find(model_name) == xclbin_map.end()) {
    return "";
  }
  return xclbin_map.at(model_name);
}

static std::string sd_get_pdi(const std::string &xclbin,
                              const std::string &op_type) {
  static std::map<std::string, SDInfoMap> xclbin_pdi_info{
      {"SD15_unet_2x4x4.xclbin",
       {
           {"SDLayerNorm", "DPU_0"},
           {"SDGemm", "DPU_0"},
           {"SDAdd", "DPU_0"},
           {"SDGelu", "DPU_0"},
           {"SDMul", "DPU_0"},
           {"SDMHA", "DPU_0"},
           {"SDConv", "DPU_1"},
           {"SDGroupNorm", "DPU_1"},
           {"SDSilu", "DPU_1"},
           {"SDConcat", "DPU_1"},
           {"SDResize", "DPU_1"},
       }},
      {"SD15_vae_2x4x4.xclbin",
       {
           {"SDGroupNorm", "DPU_0"},
           {"SDAdd", "DPU_0"},
           {"SDConv", "DPU_0"},
           {"SDSilu", "DPU_0"},
           {"SDConcat", "DPU_0"},
           {"SDResize", "DPU_0"},
           {"SDMHA", "DPU_1"},
           {"SDGemm", "DPU_2"},
       }},
      {"SD3_MMDIT_2x4x4.xclbin",
       {
           {"SDLayerNorm", "DPU_0"},
           {"SDMul", "DPU_0"},
           {"SDGelu", "DPU_0"},
           {"SDConcat", "DPU_0"},
           {"SDSlice", "DPU_0"},
           {"SDAdd", "DPU_0"},
           {"SDGemm", "DPU_0"},
           {"SDSilu", "DPU_0"},
           {"SDMHA", "DPU_0"},
           {"SDConv", "DPU_1"},
       }},
      {"SD3_1K_VAE_2x4x4.xclbin",
       {
           {"SDGroupNorm", "DPU_0"},
           {"SDAdd", "DPU_0"},
           {"SDConv", "DPU_0"},
           {"SDSilu", "DPU_0"},
           {"SDConcat", "DPU_0"},
           {"SDResize", "DPU_0"},
           {"SDMHA", "DPU_1"},
           {"SDGemm", "DPU_2"},
       }},
      {"SD3_VAE_2x4x4.xclbin",
       {
           {"SDGroupNorm", "DPU_0"},
           {"SDAdd", "DPU_0"},
           {"SDConv", "DPU_0"},
           {"SDSilu", "DPU_0"},
           {"SDConcat", "DPU_0"},
           {"SDResize", "DPU_0"},
           {"SDMHA", "DPU_1"},
           {"SDGemm", "DPU_2"},
       }},
  };
  return xclbin_pdi_info.at(xclbin).at(op_type);
}

#endif
