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

#include <array>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <vector>

namespace matmul_matrix {

#if 0
struct bfloat16_t {
  uint16_t value;
};
#endif

int constexpr Msubv = 64;
int constexpr Msubv_16 = 32;
int constexpr Ksubv = 128;
int constexpr Nsubv = 64;

int constexpr Msubv_mzdk5 = 16;
int constexpr Ksubv_mzdk5 = 80;
int constexpr Nsubv_mzdk5 = 32;
int constexpr Nsubv_mzdk5_LARGE = 128;

int constexpr Msubv_PSW_BMM = 64;
int constexpr Ksubv_PSW_BMM = 64;
int constexpr Nsubv_PSW_BMM = 32;

int constexpr Msubv_PSW = 64;
int constexpr Ksubv_PSW = 64;
int constexpr Nsubv_PSW = 48;
int constexpr Nsubv_PSW_GeMM_GeLU = 64;

int constexpr Msubv_PSW_GeMM_V_GeLU = 16;
int constexpr Ksubv_PSW_GeMM_V_GeLU = 96;
int constexpr Nsubv_PSW_GeMM_V_GeLU = 48;

int constexpr QDQparam_size = 16;
// general qdq idx
int constexpr qdq_c0_idx = 0;
int constexpr qdq_c1_idx = 2;
int constexpr qdq_c2_idx = 3;
int constexpr qdq_c3_idx = 4;
int constexpr qdq_Mv_idx = 5;
int constexpr qdq_Nv_idx = 6;
int constexpr qdq_SQb_idx = 7;
int constexpr qdq_Sout_idx = 8;
int constexpr qdq_Stdm_idx = 9;
// for mdsqr and mxpzi, set this be 1; for mxgan, if uint8_q, set it be 0, if
// uint16_q, set it be 1
int constexpr qdq_isint16_idx = 10;
int constexpr qdq_veccoeffs_idx = 10;

// gelu qdq idx
int constexpr gelu_dq_zp_idx = 0;
int constexpr gelu_dq_scale_idx = 1;
int constexpr gelu_q_zp_idx = 2;
int constexpr gelu_q_scale_idx = 3;
int constexpr gelu_isint16_idx = 4;

using SUBV_T = std::array<int, 3>;
int constexpr Mode_count = 17;
// Msubv, Ksubv, Nsubv
constexpr std::array<std::pair<int, SUBV_T>, Mode_count> subv_gemm4x4 = {
    {{0, {24, 128, 16}},
     {1, {32, 160, 16}},
     {2, {32, 160, 32}},
     {3, {16, 160, 16}},
     {4, {16, 160, 32}},
     {5, {32, 128, 64}},
     {6, {16, 128, 64}},
     {7, {32, 80, 64}},
     {8, {64, 128, 32}},
     {9, {32, 80, 80}},
     {10, {16, 192, 48}},
     {11, {8, 192, 48}},
     {12, {8, 256, 32}},
     {13, {16, 256, 32}},
     {14, {64, 64, 64}},
     {15, {8, 64, 64}},
     {16, {16, 96, 8}}}}; // PSW GemmV

int constexpr Shape_count = 36;

constexpr std::array<std::pair<SUBV_T, int>, Shape_count> subv_mode_gemm4x4 = {
    {{{77, 1024, 64}, 0},    {{96, 1024, 64}, 0},    {{4096, 320, 64}, 7},
     {{1024, 320, 640}, 2},  {{1024, 640, 64}, 5},   {{256, 640, 1280}, 8},
     {{256, 1280, 64}, 6},   {{64, 1280, 64}, 3},    {{64, 2560, 1280}, 4},
     {{256, 2560, 1280}, 8}, {{256, 1920, 1280}, 8}, {{1024, 1920, 640}, 8},
     {{1024, 1280, 640}, 8}, {{1024, 960, 640}, 2},  {{4096, 960, 320}, 9},
     {{4096, 640, 320}, 9},  {{4096, 320, 320}, 9},  {{4096, 320, 2560}, 2},
     {{4096, 1280, 320}, 9}, {{1024, 640, 640}, 8},  {{1024, 640, 5120}, 8},
     {{1024, 2560, 640}, 8}, {{256, 1280, 1280}, 8}, {{256, 1280, 10240}, 5},
     {{256, 5120, 1280}, 5}, {{64, 1280, 1280}, 6},  {{64, 1280, 10240}, 6},
     {{64, 5120, 1280}, 6},  {{1, 1280, 320}, 3},    {{1, 1280, 640}, 3},
     {{1, 1280, 1280}, 6},   {{64, 3072, 3072}, 10}, {{1, 3072, 3072}, 11},
     {{1, 8192, 3072}, 12},  {{64, 8192, 3072}, 13}, {{1, 768, 8}, 16}}};

int constexpr Shape_count_int4 = 10;

constexpr std::array<std::pair<SUBV_T, int>, Shape_count_int4>
    subv_mode_gemm4x4_int4 = {{{{64, 3072, 9216}, 14},
                               {{64, 3072, 16384}, 14},
                               {{1, 3072, 9216}, 15},
                               {{1, 3072, 16384}, 15},
                               {{64, 3072, 8192}, 14},
                               {{1, 3072, 8192}, 15},
                               {{1, 3072, 3072}, 11},
                               {{64, 3072, 3072}, 10},
                               {{1, 8192, 3072}, 12},
                               {{64, 8192, 3072}, 13}}};

constexpr bool arrayEqual(const SUBV_T &arr1, const SUBV_T &arr2) {
  for (size_t i = 0; i < 3; ++i) {
    if (arr1[i] != arr2[i]) {
      return false;
    }
  }
  return true;
}

// Function to search for a key (mode) and get subv values
constexpr SUBV_T get_subv(int &key) {
  for (size_t i = 0; i < subv_gemm4x4.size(); ++i) {
    if (subv_gemm4x4[i].first == key) {
      return subv_gemm4x4[i].second; // Key found
    }
  }
  return SUBV_T(); // Key not found
}

// Function to search for a key (shapes) and get subv mode
inline int search_subv_mode(const SUBV_T &key, int b_shift_value = 0) {
  if (b_shift_value == 0) { // int8
    for (size_t i = 0; i < subv_mode_gemm4x4.size(); ++i) {
      if (arrayEqual(subv_mode_gemm4x4[i].first, key)) {
        return subv_mode_gemm4x4[i].second; // Key found
      }
    }
  } else if (b_shift_value == 1) { // int4
    for (size_t i = 0; i < subv_mode_gemm4x4_int4.size(); ++i) {
      if (arrayEqual(subv_mode_gemm4x4_int4[i].first, key)) {
        return subv_mode_gemm4x4_int4[i].second; // Key found
      }
    }
  }
  return -1; // Key not found
}

inline int row_major_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  return (row * num_cols) + col;
}

inline int col_major_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  return (col * num_rows) + row;
}

inline int w8_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  int constexpr zz = 8;
  return (row * zz) + (col % zz) + ((col / zz) * (num_rows * zz));
}

inline int h4_index(int row, int col, int num_rows, int num_cols) {
  int constexpr zz = 4;
  return (col * zz) + (row % zz) + ((row / zz) * (zz * num_cols));
}

inline uint8_t int8_to_int4(uint8_t in_val, int idx) {
  return ((idx == 1) ? in_val >> 4 : (in_val & 0x0f));
}

inline uint8_t int4_pack_to_int8(uint8_t in_l, uint8_t in_h, int is_odd) {
  uint8_t out_val;
  if (is_odd == 0) {
    out_val = (in_l & 0x0F) + ((in_h & 0x0F) << 4);
  } else {
    out_val = (in_l >> 4) + ((in_h >> 4) << 4);
  }
  return (out_val);
}

template <typename T, int subv_rows, int subv_cols> struct ActMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  ActMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    // assert(num_rows % subv_rows == 0);
    // assert(num_cols % subv_cols == 0);
  }

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const idx = row * num_cols + col;
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T> struct WgtMatrix {
  int const num_rows;
  int const num_cols;
  int const subv_rows;
  int const subv_cols;
  T *const data;

  WgtMatrix(int num_rows, int num_cols, int subv_rows, int subv_cols,
            void *data)
      : num_rows(num_rows), num_cols(num_cols), subv_rows(subv_rows),
        subv_cols(subv_cols), data(static_cast<T *>(data)) {
    assert(num_rows % subv_rows == 0);
    assert(num_cols % subv_cols == 0);
  }

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const subv_size = subv_rows * subv_cols;
    int const r = row % subv_rows;
    int const c = col % subv_cols;
    int const i = w8_index(r, c, subv_rows, subv_cols);
    int const rr = row / subv_rows;
    int const cc = col / subv_cols;
    int const ii =
        col_major_index(rr, cc, (num_rows / subv_rows), (num_cols / subv_cols));
    int const idx = i + (ii * subv_size);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

struct int4_WgtMatrix {
  int const num_rows;
  int const num_cols;
  int const subv_rows;
  int const subv_cols;
  uint8_t *const data;

  int4_WgtMatrix(int num_rows, int num_cols, int subv_rows, int subv_cols,
                 void *data)
      : num_rows(num_rows), num_cols(num_cols), subv_rows(subv_rows),
        subv_cols(subv_cols), data(static_cast<uint8_t *>(data)) {
    assert(num_rows % subv_rows == 0);
    assert(num_cols % subv_cols == 0);
  }

  uint8_t &block(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const subv_size = subv_rows * subv_cols / 2;
    int const r = row % subv_rows;
    int const c = col % subv_cols;
    int const i = w8_index(r, c, subv_rows, subv_cols) / 2;
    int const rr = row / subv_rows;
    int const cc = col / subv_cols;
    int const ii =
        col_major_index(rr, cc, (num_rows / subv_rows), (num_cols / subv_cols));
    int const idx = i + (ii * subv_size);
    // printf("num_rows = %d, num_cols = %d, row = %d, col = %d , i = %d, idx =
    // %d \n", num_rows, num_cols, row, col, i, idx);
    assert(idx < (num_rows * num_cols / 2));
    return data[idx];
  }

  uint8_t at(int row, int col) {
    return (int8_to_int4(block(row, col), (col % 2)));
  }

  static int size(int num_rows, int num_cols) {
    return (num_rows * num_cols * sizeof(uint8_t) / 2);
  }
};

template <typename T, int subv_rows, int subv_cols, int aie_rows = 4,
          int aie_cols = 2>
struct BiaVector {
  int const num_rows;
  int const num_cols;
  T *const data;

  BiaVector(int num_cols, void *data)
      : num_rows(1), num_cols(num_cols), data(static_cast<T *>(data)) {}

  T &at(int row, int col) {
    int const idx = col + row * num_cols;
    return data[idx];
  }

  static int size(int num_cols) { return num_cols * sizeof(T); }
};

template <typename T, int subv_rows, int subv_cols, int aie_rows = 4,
          int aie_cols = 2>
struct OutMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  OutMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    // assert(num_rows % (subv_rows * aie_rows) == 0);
    //  assert(num_cols % (subv_cols * aie_cols) == 0);
  }

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const idx = row * num_cols + col;
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T> struct RowMajorMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  RowMajorMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {}

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int idx = row_major_index(row, col, num_rows, num_cols);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T, int subv_rows, int subv_cols> struct BiasVector {
  int const num_rows;
  int const num_cols;
  T *const data;

  BiasVector(int num_cols, void *data)
      : num_rows(1), num_cols(num_cols), data(static_cast<T *>(data)) {}

  T &at(int row, int col) {
    int const idx = col + row * num_cols;
    return data[idx];
  }

  static int size(int num_cols) { return num_cols * sizeof(T); }
};

template <typename T>
void format_wgt_trans(T *wgt_data, T *buf, int subv_mode, int K, int N,
                      int K_orig, int b_shift_value) {
  SUBV_T subv = get_subv(subv_mode);
  int Ksv = subv[1];
  int Nsv = subv[2];
  if (b_shift_value == 0) {
    matmul_matrix::WgtMatrix<T> W(K, N, Ksv, Nsv, buf);
    for (int r = 0; r < K_orig; ++r) {
      for (int c = 0; c < N; ++c) {
        W.at(r, c) = wgt_data[(c * K_orig) + r]; // transpose
      }
    }
  } else if (b_shift_value == 1) { // int4
    matmul_matrix::int4_WgtMatrix W(K, N, Ksv, Nsv, buf);
    for (int r = 0; r < K_orig; r += 2) {
      int is_odd_0 = r % 2;
      int is_odd_1 = (r + 1) % 2;
      for (int c = 0; c < N; c += 2) {
        uint8_t temp = int4_pack_to_int8(
            wgt_data[(c * K_orig / 2) + r / 2],
            wgt_data[((c + 1) * K_orig / 2) + r / 2], is_odd_0);
        W.block(r, c) = temp; // transpose
        temp = int4_pack_to_int8(wgt_data[(c * K_orig / 2) + r / 2],
                                 wgt_data[((c + 1) * K_orig / 2) + r / 2],
                                 is_odd_1);
        W.block(r + 1, c) = temp; // transpose
      }
    }
  } else {
    throw std::runtime_error("Conv2Matmul : Invalid b_shift_value");
  }
}

template <typename T>
void format_wgt(T *wgt_data, T *buf, int subv_mode, int K, int N) {
  SUBV_T subv = get_subv(subv_mode);
  int Ksv = subv[1];
  int Nsv = subv[2];
  matmul_matrix::WgtMatrix<T> W(K, N, Ksv, Nsv, buf);
  for (int r = 0; r < K; ++r) {
    for (int c = 0; c < N; ++c) {
      W.at(r, c) = wgt_data[(r * N) + c];
    }
  }
}

template <typename T> void init_random(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      mat.at(i, j) = (rand() % (max - min)) + min;
    }
  }
}

inline float gelu_golden(float in) {
  float xr2 = in / (std::sqrt(2.0f));
  float t = std::erf(xr2);
  float g = in * 0.5f * (1.0f + t);

  return g;
}

inline float silu_golden(float in) {
  float exp_x = std::exp(-in);
  float sg = in / (1 + exp_x);

  return sg;
}

inline float sigmoid_golden(float in, int r, int c) {
  float exp_x = std::exp(-in);
  float sg = 1 / (1 + exp_x);

  return sg;
}

inline uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *tmp = (uint8_t *)&i;
  // copy float to uint32_t
  tmp[0] = src[0];
  tmp[1] = src[1];
  tmp[2] = src[2];
  tmp[3] = src[3];
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

template <typename T1, typename T2>
void dequant_to_bfloat(std::vector<T1> &in_vec, std::vector<T2> &out_vec,
                       int zero_point, float scale) {
  auto num_cols = in_vec.size();
  for (int i = 0; i < num_cols; ++i) {
    out_vec[i] = float_to_bfloat16((in_vec[i] - zero_point) * scale);
  }
}

template <typename T1, typename T2>
void dequant_int8_to_bfloat(std::vector<T1> &in_vec, std::vector<T2> &out_vec,
                            uint8_t zero_point, float scale) {
  auto num_cols = in_vec.size();
  for (int i = 0; i < num_cols; ++i) {
    out_vec[i] = float_to_bfloat16((in_vec[i] - zero_point) * scale);
  }
}
#if 1
inline float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}
#else
inline float uint_to_float(uint32_t i) {
  float f = 0;
  char *ptr_f = reinterpret_cast<char *>(&f);
  char *ptr_i = reinterpret_cast<char *>(&i);
  ptr_f[0] = ptr_i[0];
  ptr_f[1] = ptr_i[1];
  ptr_f[2] = ptr_i[2];
  ptr_f[3] = ptr_i[3];
  return f;
}

inline float bfloat16_to_float(bfloat16_t bf) {
  return uint_to_float(uint32_t(bf.value) << 16);
}
#endif

union Float32Bits {
  uint32_t u;
  float f;
};

static const uint32_t kF32BfMantiBitDiff = 16;

static float bfloat2float(uint16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << kF32BfMantiBitDiff;
  return floatBits.f;
}

static int16_t float2bfloat(float floatValue) {
  if (std::isnan(floatValue)) {
    return std::signbit(floatValue) ? 0xFFC0 : 0x7FC0;
  }

  Float32Bits floatBits;
  floatBits.f = floatValue;
  uint16_t bfloatBits;

  // Least significant bit of resulting bfloat.
  uint32_t lsb = (floatBits.u >> kF32BfMantiBitDiff) & 1;
  uint32_t roundingBias = 0x7fff + lsb;
  floatBits.u += roundingBias;
  bfloatBits = static_cast<int16_t>(floatBits.u >> kF32BfMantiBitDiff);
  return bfloatBits;
}

template <typename T>
void initialize_random_bfloat16(std::vector<T> &vec, size_t size,
                                float data_min, float data_max) {
  for (int i = 0; i < size; i++) {
    vec[i] = float2bfloat(
        float((rand() / (RAND_MAX / (data_max - data_min))) + data_min));
  }
}

template <typename T> T saturate(T val, T min, T max) {
  return std::min(std::max(val, min), max);
}

template <typename T> inline T srs_to_int8(int64_t x, int shift = 0) {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<int8_t>(
        saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, -128, 127));
  } else {
    int64_t inp_floor = (x >> shift);
    int64_t inp_frac = x - (inp_floor << shift);
    if (inp_frac == (1 << (shift - 1))) {
      if (inp_floor % 2) { // odd
        return static_cast<uint8_t>(saturate<int32_t>(inp_floor + 1, 0, 255));
      } else {
        return static_cast<uint8_t>(saturate<int32_t>(inp_floor, 0, 255));
      }
    } else {
      return static_cast<uint8_t>(
          saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, 0, 255));
    }
  }
}

template <typename T> inline T srs_to_int16(int64_t x, int shift = 0) {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<int16_t>(
        saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, INT16_MIN, INT16_MAX));
  } else {
    int64_t inp_floor = (x >> shift);
    int64_t inp_frac = x - (inp_floor << shift);
    if (inp_frac == (1 << (shift - 1))) {
      if (inp_floor % 2) { // odd
        return static_cast<uint16_t>(
            saturate<uint32_t>(inp_floor + 1, 0, UINT16_MAX));
      } else {
        return static_cast<uint16_t>(
            saturate<uint32_t>(inp_floor, 0, UINT16_MAX));
      }
    } else {
      return static_cast<uint16_t>(
          saturate<uint32_t>(((x >> (shift - 1)) + 1) >> 1, 0, UINT16_MAX));
    }
  }
}

inline int32_t srs_to_int32(int64_t x, int shift, bool sign = true) {
  if (sign) {
    return static_cast<int32_t>(saturate<int32_t>(
        int32_t(((x >> (shift - 1)) + 1) >> 1), INT32_MIN, INT32_MAX));
  } else {
    int64_t inp_floor = (x >> shift);
    int64_t inp_frac = x - (inp_floor << shift);
    if (inp_frac == (int64_t(1) << (shift - 1))) {
      if (inp_floor % 2) { // odd
        return static_cast<uint32_t>(
            saturate<uint32_t>(uint32_t(inp_floor + 1), 0, UINT32_MAX));
      } else {
        return static_cast<uint32_t>(
            saturate<uint32_t>(uint32_t(inp_floor), 0, UINT32_MAX));
      }
    } else {
      return static_cast<uint32_t>(saturate<uint32_t>(
          uint32_t(((x >> (shift - 1)) + 1) >> 1), 0, UINT32_MAX));
    }
  }
}

inline int64_t sls_to_int64(int64_t x, int shift, bool sign = true) {
  // NOTE: No rounding when upshifted
  if (sign) {
    return static_cast<int64_t>(x << shift);
  } else {
    return static_cast<uint64_t>(x << shift);
  }
}

template <typename Ta, typename Toi, typename Tout>
void qdq_golden(Ta A, Toi X, int32_t C2, int32_t C1, int64_t *C0, uint8_t sqb,
                uint8_t sout, Tout Y, std::string Ytype) {
  // uint32_t ifmsum[A.num_rows];
  std::vector<int64_t> ifmsum;
  for (int r = 0; r < A.num_rows; ++r) {
    int64_t acc = 0;
    for (int c = 0; c < A.num_cols; ++c) {
      acc += A.at(r, c);
      // printf("r:%d c:%d, A.at(r,c) = %d, ifmsum[r] = %d\n", r, c,A.at(r,c),
      // ifmsum[r]);
    }
    ifmsum.push_back(acc);
  }

  for (int c = 0; c < X.num_cols; ++c) {
    for (int r = 0; r < X.num_rows; ++r) {
      if (Ytype == "uint8") {
        Y.at(r, c) = srs_to_int8<std::uint8_t>(
            (int64_t)X.at(r, c) * (int64_t)C2 +
                (int64_t)C1 * (int64_t)ifmsum[r] + ((int64_t)C0[c] << sqb),
            sout);
      } else { // uint16
        Y.at(r, c) = srs_to_int16<std::uint16_t>(
            (int64_t)X.at(r, c) * (int64_t)C2 +
                (int64_t)C1 * (int64_t)ifmsum[r] + ((int64_t)C0[c] << sqb),
            sout);
      }
      // if ((r == 0 && c == 0) || (r == 0 && c == 1)) {
      //    printf("r:%d, c:%d, X=%d, C2=%d, C1=%d, ifmsum[r]=%d\n", r, c,
      //    (int64_t)X.at(r, c), (int64_t)C2, (int64_t)C1, (int64_t)ifmsum[r]);
      //    printf("r:%d, c:%d, C0[c]:%lld, sqb:%d, sout:%d, X*+: %lld,
      //    shifted:%d\n", r, c, (int64_t)C0[c], sqb, sout, (int64_t)X.at(r, c)
      //    *
      //        (int64_t)C2 + (int64_t)C1 * (int64_t)ifmsum[r] + ((int64_t)C0[c]
      //        << sqb), (int64_t)((int64_t)X.at(r, c)
      //    * (int64_t)C2 + (int64_t)C1 * (int64_t)ifmsum[r] + ((int64_t)C0[c]
      //    << sqb)) >> sout);
      // }
    }
  }
}

template <typename Ta, typename Toi, typename Tout>
void qdq_golden(Ta A, Toi X, int32_t *C2_vec, int32_t *C1_vec, int64_t *C0,
                uint8_t sqb, uint8_t sout, Tout Y, std::string Ytype) {
  std::vector<int64_t> ifmsum;
  for (int r = 0; r < A.num_rows; ++r) {
    int64_t acc = 0;
    for (int c = 0; c < A.num_cols; ++c) {
      acc += A.at(r, c);
      // printf("r:%d c:%d, A.at(r,c) = %d, ifmsum[r] = %d\n", r, c,A.at(r,c),
      // ifmsum[r]);
    }
    ifmsum.push_back(acc);
  }

  for (int c = 0; c < X.num_cols; ++c) {
    for (int r = 0; r < X.num_rows; ++r) {
      if (Ytype == "uint8") {
        Y.at(r, c) = srs_to_int8<std::uint8_t>(
            (int64_t)X.at(r, c) * (int64_t)C2_vec[c] +
                (int64_t)C1_vec[c] * (int64_t)ifmsum[r] +
                ((int64_t)C0[c] << sqb),
            sout);
      } else { // uint16
        Y.at(r, c) = srs_to_int16<std::uint16_t>(
            (int64_t)X.at(r, c) * (int64_t)C2_vec[c] +
                (int64_t)C1_vec[c] * (int64_t)ifmsum[r] +
                ((int64_t)C0[c] << sqb),
            sout);
      }
    }
  }
}

template <typename T1, typename T2, typename T3, typename T4>
void qdq_asym_golden(T1 A, T2 B, T3 X, int32_t C0, int32_t C2, int32_t C1,
                     int32_t C3, uint8_t sqb, uint8_t sout, T4 Y) {
  // int32_t ifmsum[A.num_rows];
  std::vector<int32_t> ifmsum;
  for (int r = 0; r < A.num_rows; ++r) {
    int32_t acc = 0;
    for (int c = 0; c < A.num_cols; ++c) {
      acc += A.at(r, c);
      // printf("r:%d c:%d, A.at(r,c) = %d, ifmsum[r] = %d\n", r, c,A.at(r,c),
      // ifmsum[r]);
    }
    ifmsum.push_back(acc);
  }
  // int32_t wgtsum[B.num_cols];
  std::vector<int32_t> wgtsum;
  for (int c = 0; c < B.num_cols; ++c) {
    int32_t acc = 0;
    for (int r = 0; r < B.num_rows; ++r) {
      acc += B.at(r, c);
    }
    // printf("c:%d, wgtsum[r] = %d\n", c, wgtsum[c]);
    wgtsum.push_back(acc);
  }

  for (int c = 0; c < X.num_cols; ++c) {
    for (int r = 0; r < X.num_rows; ++r) {
      Y.at(r, c) = srs_to_int8<std::int8_t>(X.at(r, c) * C2 + C1 * ifmsum[r] +
                                                ((wgtsum[c] * C0 + C3) >> sqb),
                                            sout);
      // printf("r:%d, c:%d, (int64_t)X.at(r, c)=%d, (int64_t)C1 *
      // (int64_t)ifmsum[r] =%d, ((int64_t)C0[c] << sqb)=%d, C2=%d,
      // (int64_t)ifmsum[r]=%d\n", r,c, (int64_t)X.at(r, c), (int64_t)C1 *
      // (int64_t)ifmsum[r]  , ((int64_t)C0[c] << sqb), C2, (int64_t)ifmsum[r]);
      // printf("r:%d, c:%d, X:%d, C1:%d, C0[c]:%d, sqb:%d, sout:%d, X*+: %d,
      // shifted:%d\n", r,c,X.at(r, c),C1,C0[c],sqb,sout,X.at(r, c) * C1 +
      // (C0[r] << sqb), (X.at(r, c) * C1 + (C0[r] << sqb))>>sout);
    }
  }
}

static uint16_t srs_to_uint16(int32_t x) {
  return static_cast<uint16_t>(saturate<int32_t>(x, 0, UINT16_MAX));
}

template <typename T1, typename T2, typename T3>
void quant(T1 mat, T2 Out, float s, T3 z, std::string Ytype) {
  if (Ytype == "uint8") {
    for (int i = 0; i < mat.num_rows; ++i) {
      for (int j = 0; j < mat.num_cols; ++j) {
        int32_t temp =
            static_cast<int>(bfloat16_to_float(mat.at(i, j)) / s + z);
        Out.at(i, j) = srs_to_int8<std::uint8_t>(temp);
      }
    }
  } else {
    for (int i = 0; i < mat.num_rows; ++i) {
      for (int j = 0; j < mat.num_cols; ++j) {
        Out.at(i, j) = static_cast<uint16_t>(saturate<int32_t>(
            (std::round(bfloat16_to_float(mat.at(i, j)) / s) + z), 0, 65535));
      }
    }
  }
}
template <typename T1, typename T2, typename T3>
void quant_bfloat16_to_int16(T1 mat, T2 Out, float inv_s, T3 z) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      Out.at(i, j) = static_cast<int16_t>(
          std::round(bfloat16_to_float(mat.at(i, j)) * inv_s) + z);
    }
  }
}

template <typename T>
void quant_bfloat_to_uint16(T in_mat, int16_t sc_out, uint16_t zp_out,
                            T out_mat) {
  float sc = bfloat16_to_float(sc_out);
  for (int i = 0; i < in_mat.num_rows; ++i) {
    for (int j = 0; j < in_mat.num_cols; ++j) {
      int32_t temp =
          static_cast<int>(bfloat16_to_float(in_mat.at(i, j)) * sc + zp_out);
      out_mat.at(i, j) = srs_to_uint16(temp);
    }
  }
}
template <typename Tx, typename Tw, typename Ty>
void cpu_matmul(Tx X, Tw W, Ty Y, std::string Ytype) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      uint32_t acc = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      if (Ytype == "int16") {
        Y.at(r, c) = srs_to_int16<std::int16_t>(acc);
      } else if (Ytype == "int8") {
        Y.at(r, c) = srs_to_int8<std::int8_t>(acc, 12);
      } else if (Ytype == "int32") {
        Y.at(r, c) = acc;
      }
    }
  }
}

template <typename Tx, typename Tw, typename Tb, typename Ty>
void cpu_matmul(Tx X, Tw W, Tb B, Ty Y, std::string Ytype) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int32_t acc = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      if (Ytype == "int16") {
        Y.at(r, c) = srs_to_int16<std::int16_t>(acc) + B.at(0, c);
      } else if (Ytype == "int8") {
        Y.at(r, c) = srs_to_int8<std::int8_t>(acc, 12) + B.at(0, c);
      } else if (Ytype == "int32") {
        Y.at(r, c) = acc;
      }
    }
  }
}

template <typename Ta, typename Tw, typename To>
void cpu_matmul(Ta X, Tw W, To Y, int shift, int Msubv, int Ksubv, int Nsubv) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int64_t acc = 0;
      for (int k_shard = 0; k_shard < (X.num_cols) / Ksubv; ++k_shard) {
        /*
         *  Presently the AIE GEMM kernel does upshift before loading the acc
         *  Followed by down shift when writing back the acc to TDM
         *  This is done at the subvol boundary (32 x 128 x 64)
         *  For this reason, the inner dim is broken down to shards of 128
         *  and encapsulated by up/down shift as seen below
         */
        acc = sls_to_int64(acc, shift, false);
        for (int k = 0; k < Ksubv; ++k) {
          acc += (X.at(r, ((k_shard * Ksubv) + k)) *
                  W.at(((k_shard * Ksubv) + k), c));
        }
        acc = srs_to_int32(acc, shift, false);
      }
      Y.at(r, c) = acc;
    }
  }
}

template <typename Ta, typename Tw, typename To>
void cpu_matmul(Ta X, Tw W, To Y, int shift) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int64_t acc = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      acc = srs_to_int32(acc, shift, false);
      Y.at(r, c) = acc;
    }
  }
}

/*
 * CPU GEMM: (X * W) + B
 */
template <typename Tx, typename Tw, typename Tb, typename Ty>
void cpu_matmul_bias(Tx X, Tw W, Tb B, Ty Y, std::string Ytype) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int32_t acc = B.at(0, c);
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      if (Ytype == "int16") {
        Y.at(r, c) = srs_to_int16<std::int16_t>(acc);
      } else if (Ytype == "int8") {
        Y.at(r, c) = srs_to_int8<std::int8_t>(acc, 12);
      }
    }
  }
}

static int const frac_bits = 10;

template <typename T>
int check_result(T cpu_Y, T aie_Y, bool enable_logging = false) {
  int fail = 0;
  int err_count = 0;
  int max_diff = 0;
  float L2_norm = 0;
  for (int r = 0; r < aie_Y.num_rows; ++r) {
    for (int c = 0; c < aie_Y.num_cols; ++c) {
      int32_t diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > 1) {
        if (enable_logging) {
          std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                    << "Expected: " << int(cpu_Y.at(r, c)) << ", "
                    << "Received: " << int(aie_Y.at(r, c)) << ", "
                    << "Diff: " << int(diff) << "\n";
        }
        fail = 1;
        err_count++;
      }
    }
  }
  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

template <typename T>
int check_add_result(T cpu_Y, T aie_Y, float error_tolerance = 0.01) {
  int fail = 0;
  int err_count = 0;
  int max_diff = 0;
  float L2_norm = 0;
  for (int r = 0; r < aie_Y.num_rows; ++r) {
    for (int c = 0; c < aie_Y.num_cols; ++c) {
      int32_t diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      float relative_diff = diff / (cpu_Y.at(r, c) + 0.0001);
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (relative_diff > error_tolerance) {
        std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                  << "Expected: " << int(cpu_Y.at(r, c)) << ", "
                  << "Received: " << int(aie_Y.at(r, c)) << ", "
                  << "Diff: " << int(diff) << "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

template <typename To>
int check_result_uint16(To cpu_Y, To aie_Y, float error_tolerance = 0.01) {
  int fail = 0;
  uint16_t max_diff = 0;
  float L2_norm = 0;
  int err_count = 0;
  for (int r = 0; r < cpu_Y.num_rows; ++r) {
    for (int c = 0; c < cpu_Y.num_cols; ++c) {
      uint16_t diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if ((float)diff > error_tolerance) {
        std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                  << "Expected: " << cpu_Y.at(r, c) << ","
                  << "Received: " << aie_Y.at(r, c) << "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

template <typename To>
int check_result_bfloat(To cpu_Y, To aie_Y, float error_tolerance = 0.01) {
  int fail = 0;
  float max_diff = 0;
  float L2_norm = 0;
  int err_count = 0;
  for (int r = 0; r < cpu_Y.num_rows; ++r) {
    for (int c = 0; c < cpu_Y.num_cols; ++c) {
      float diff = std::abs(bfloat16_to_float(cpu_Y.at(r, c)) -
                            bfloat16_to_float(aie_Y.at(r, c)));
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > error_tolerance) {
        // std::cout << "ERROR: Y[" << r << ", " << c << "]: "
        //           << "Expected: " << bfloat16_to_float(cpu_Y.at(r, c)) << ","
        //           << "Received: " << bfloat16_to_float(aie_Y.at(r, c)) <<
        //           "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

template <typename T>
int check_add_result_bfloat16(std::vector<T> cpu_Y, std::vector<T> aie_Y,
                              std::vector<size_t> tensor_shape,
                              float error_tolerance = 0.01) {
  auto num_rows = tensor_shape[0];
  auto num_cols = tensor_shape[1];

  int fail = 0;
  int err_count = 0;
  float max_diff = 0.0;
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      float cpu_out_float = bfloat16_to_float(cpu_Y.at(r * num_cols + c));
      float aie_out_float = bfloat16_to_float(aie_Y.at(r * num_cols + c));
      float diff = std::abs(cpu_out_float - aie_out_float);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > error_tolerance) {
        // printf("ERROR: Y[%d][%d] Expected: %f, %d, Received: %f, %d \n", r,
        // c,
        //        bfloat16_to_float(cpu_Y.at(r * num_cols + c)), cpu_Y.at(r *
        //        num_cols + c), bfloat16_to_float(aie_Y.at(r * num_cols + c)),
        //        aie_Y.at(r * num_cols + c));
        // std::cout << std::fixed << std::setprecision(6)
        //          << "golden: " << cpu_out_float
        //          << " | "
        //          << "Output: " << aie_out_float
        //          << " | "
        //          << "Error = " << diff << std::endl;
        fail = 1;
        err_count++;
      } else {
        // std::cout << std::fixed << std::setprecision(6)
        //           << "golden: " << cpu_out_float
        //           << " | "
        //           << "Output: " << aie_out_float
        //           << " | "
        //           << "Error = " << diff << std::endl;
      }
    }
  }
  std::cout << "max_diff is " << max_diff << std::endl;
  return fail;
}

} // namespace matmul_matrix
