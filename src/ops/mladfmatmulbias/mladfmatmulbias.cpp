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

#include <any>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>
// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ops/op_interface.hpp>
#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>
#include <xclbin_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include "ops/ops_common/mladf_matmul_matrix.hpp"
#include <txn_helper/txn_tiling_util.hpp>

namespace ryzenai {

namespace {
std::string getXCLBinName(std::string op_version) {
  if (op_version == "v1") {
    return LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else if (op_version == "flat") {
    return LLAMA2_MLADF_2x4x4_BFP16_GEMM_SILU_MUL_FLAT_RMS_XCLBIN_NAME;
  } else if (op_version == "bfp16gemm") {
    return "stx_llama2_mladf_2x4x4_bfp16gemm_Bfp16GemmCast";
  } else if (op_version == "v2") {
    return LLAMA2_MLADF_2x4x4_V2_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else {
    return LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  }
}
} // namespace

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::string mladfmatmulbias<InT, WtT, AccT, OutT>::get_instr_key(
    std::string prefix, size_t m, size_t k, size_t n, size_t grp_size) const {
  if (grp_size) {
    return "mladfmatmulbias_" + prefix + "_" + std::to_string(m) + "_" +
           std::to_string(k) + "_" + std::to_string(n) + "_" +
           std::to_string(grp_size);
  } else {
    return "mladfmatmulbias_" + prefix + "_" + std::to_string(m) + "_" +
           std::to_string(k) + "_" + std::to_string(n);
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<mladf_matrix_shapes> &
mladfmatmulbias<InT, WtT, AccT, OutT>::get_supported_shapes() {
  return supported_shapes_;
}

std::vector<std::vector<int32_t>> f2bfp(const std::vector<float> &f,
                                        int ebs = 8) {
  const int bits = 16;
  const int dim = -1;
  const bool m2_0 = false;
  const bool m2_0_rnd = false;

  size_t num_elements = f.size();
  assert(num_elements % ebs == 0);

  size_t reshaped_dim1 = num_elements / ebs;
  std::vector<std::vector<int32_t>> reshaped_f(reshaped_dim1,
                                               std::vector<int32_t>(ebs, 0));

  for (size_t i = 0; i < reshaped_dim1; ++i) {
    for (size_t j = 0; j < ebs; ++j) {
      reshaped_f[i][j] = *reinterpret_cast<const int32_t *>(&f[i * ebs + j]);
    }
  }

  std::vector<std::vector<int32_t>> fis(reshaped_dim1,
                                        std::vector<int32_t>(ebs, 0));
  std::vector<std::vector<int32_t>> fie(reshaped_dim1,
                                        std::vector<int32_t>(ebs, 0));
  std::vector<std::vector<int32_t>> fim(reshaped_dim1,
                                        std::vector<int32_t>(ebs, 0));

  for (size_t i = 0; i < reshaped_dim1; ++i) {
    for (size_t j = 0; j < ebs; ++j) {
      int32_t value = reshaped_f[i][j];
      fis[i][j] = (value < 0) ? -1 : 1;
      fie[i][j] = (value >> 23) & 255;
      fim[i][j] = ((fie[i][j] > 0) << 23) | (value & ((1 << 23) - 1));
    }
  }

  std::vector<int32_t> me(reshaped_dim1, 0);
  for (size_t i = 0; i < reshaped_dim1; ++i) {
    for (size_t j = 0; j < ebs; ++j) {
      me[i] = std::max(me[i],
                       fie[i][j] - static_cast<int>(m2_0) *
                                       (fis[i][j] * fim[i][j] == (-1 << 23)));
    }
  }

  std::vector<std::vector<int32_t>> shift(reshaped_dim1,
                                          std::vector<int32_t>(ebs, 0));
  std::vector<std::vector<int32_t>> bm(reshaped_dim1,
                                       std::vector<int32_t>(ebs, 0));
  int32_t bm_max = (1 << (bits - 9)) - 1;

  for (size_t i = 0; i < reshaped_dim1; ++i) {
    for (size_t j = 0; j < ebs; ++j) {
      shift[i][j] = me[i] - fie[i][j] + 33 - bits;
      int32_t test_im =
          fim[i][j] +
          ((shift[i][j] > 0) << std::min(23, std::max(0, shift[i][j] - 1)));
      int32_t test_bm =
          (test_im >> std::min(32, shift[i][j])) * (m2_0_rnd ? fis[i][j] : 1);
      bool overflow = test_bm > bm_max;
      if (overflow) {
        shift[i][j]++;
        me[i]++;
      }
      fim[i][j] +=
          ((shift[i][j] > 0) << std::min(23, std::max(0, shift[i][j] - 1)));
      bm[i][j] = (fim[i][j] >> std::min(32, shift[i][j])) * fis[i][j];
    }
  }

  std::vector<std::vector<int32_t>> output_bfp16(
      reshaped_dim1, std::vector<int32_t>(ebs + 1, 0));
  for (size_t i = 0; i < reshaped_dim1; ++i) {
    output_bfp16[i][0] = me[i] - 256 * (me[i] >= 128);
    for (size_t j = 0; j < ebs; ++j) {
      output_bfp16[i][j + 1] = bm[i][j];
    }
  }

  return output_bfp16;
}

// Compress int4 data into int16
std::vector<uint16_t> compress_int4_to_int16(const std::vector<int8_t> &data,
                                             const std::vector<size_t> &shape) {
  assert(!shape.empty() && shape.back() % 4 == 0 &&
         "Innermost dimension must be a multiple of 4 for compression.");

  std::vector<size_t> compressed_shape = shape;
  compressed_shape.back() /= 4;

  size_t original_size = 1, compressed_size = 1;
  for (size_t dim : shape) {
    original_size *= dim;
  }
  for (size_t dim : compressed_shape) {
    compressed_size *= dim;
  }

  std::vector<uint16_t> compressed(compressed_size, 0);

  size_t inner_dim = shape.back();
  size_t outer_size = original_size / inner_dim;
  size_t compressed_inner_dim = compressed_shape.back();

  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    for (size_t i = 0; i < compressed_inner_dim; ++i) {
      size_t base_idx = outer_idx * inner_dim + i * 4;
      uint16_t value = 0;
      value |= static_cast<uint16_t>(data[base_idx] & 0xF);
      value |= static_cast<uint16_t>(data[base_idx + 1] & 0xF) << 4;
      value |= static_cast<uint16_t>(data[base_idx + 2] & 0xF) << 8;
      value |= static_cast<uint16_t>(data[base_idx + 3] & 0xF) << 12;
      compressed[outer_idx * compressed_inner_dim + i] = value;
    }
  }

  return compressed;
}

static std::vector<uint16_t> aie_srs(const std::vector<uint32_t> &input,
                                     const std::vector<int> &aie_mode) {
  // Assign AIE mode parameters
  int data_width = aie_mode[0];
  int shift = aie_mode[1];
  int mode = aie_mode[2];
  bool is_unsigned = (aie_mode[3] != 0);

  // Output vector
  std::vector<uint16_t> output(input.size());

  // Shift and process each input element
  for (size_t i = 0; i < input.size(); ++i) {
    double shifted_value = input[i] / static_cast<double>(1 << shift);

    // Rounding
    if (mode == 0) {
      // Simply truncate (floor operation)
      shifted_value = std::floor(shifted_value);
    } else if (mode == 2) {
      // Round to nearest, half up
      shifted_value = std::round(shifted_value);
      if ((input[i] % (1 << shift)) == (1 << (shift - 1)) &&
          shifted_value < 0) {
        shifted_value = std::round(input[i] + 0.5) / (1 << shift);
      }
    } else if (mode == 6) {
      // Round to nearest, half to even
      shifted_value = std::nearbyint(shifted_value);
    } else {
      std::cerr << "Unexpected rounding mode" << std::endl;
      return {};
    }

    // Saturation
    if (is_unsigned) {
      shifted_value = std::min(std::max(shifted_value, 0.0),
                               static_cast<double>((1 << data_width) - 1));
    } else {
      shifted_value =
          std::min(std::max(shifted_value, -std::pow(2, data_width - 1)),
                   std::pow(2, data_width - 1) - 1);
    }

    // Store as uint16_t
    output[i] = static_cast<uint16_t>(shifted_value);
  }

  return output;
}

std::vector<uint16_t> concatenate_along_axis3(
    const std::vector<uint16_t> &wts_int16, size_t wts_dim0, size_t wts_dim1,
    size_t wts_dim2, size_t wts_dim3, const std::vector<uint16_t> &zp_int16,
    size_t zp_dim0, size_t zp_dim1, size_t zp_dim2, size_t zp_dim3,
    const std::vector<uint16_t> &scale_int16, size_t scale_dim0,
    size_t scale_dim1, size_t scale_dim2, size_t scale_dim3) {
  // Check dimensions for compatibility
  if (wts_dim0 != zp_dim0 || wts_dim0 != scale_dim0 || wts_dim1 != zp_dim1 ||
      wts_dim1 != scale_dim1 || wts_dim2 != zp_dim2 || wts_dim2 != scale_dim2) {
    throw std::invalid_argument(
        "Input dimensions are not compatible for concatenation.");
  }

  // Verify size matches the dimensions
  size_t expected_wts_size =
      static_cast<size_t>(wts_dim0) * wts_dim1 * wts_dim2 * wts_dim3;
  size_t expected_zp_size =
      static_cast<size_t>(zp_dim0) * zp_dim1 * zp_dim2 * zp_dim3;
  size_t expected_scale_size =
      static_cast<size_t>(scale_dim0) * scale_dim1 * scale_dim2 * scale_dim3;

  if (wts_int16.size() != expected_wts_size) {
    throw std::invalid_argument("Mismatch in wts_int16 size. Expected: " +
                                std::to_string(expected_wts_size) +
                                ", Got: " + std::to_string(wts_int16.size()));
  }

  if (zp_int16.size() != expected_zp_size) {
    throw std::invalid_argument("Mismatch in zp_int16 size. Expected: " +
                                std::to_string(expected_zp_size) +
                                ", Got: " + std::to_string(zp_int16.size()));
  }

  if (scale_int16.size() != expected_scale_size) {
    throw std::invalid_argument("Mismatch in scale_int16 size. Expected: " +
                                std::to_string(expected_scale_size) +
                                ", Got: " + std::to_string(scale_int16.size()));
  }

  // New dimensions after concatenation
  int combined_dim0 = static_cast<int>(wts_dim0);
  int combined_dim1 = static_cast<int>(wts_dim1);
  int combined_dim2 = static_cast<int>(wts_dim2);
  int combined_dim3 = static_cast<int>(wts_dim3 + zp_dim3 + scale_dim3);

  // Create output array
  std::vector<uint16_t> combined_data(combined_dim0 * combined_dim1 *
                                      combined_dim2 * combined_dim3);

  // Fill the combined array
  for (int i = 0; i < combined_dim0; ++i) {
    for (int j = 0; j < combined_dim1; ++j) {
      for (int k = 0; k < combined_dim2; ++k) {
        size_t base_offset =
            (i * combined_dim1 * combined_dim2 * combined_dim3) +
            (j * combined_dim2 * combined_dim3) + (k * combined_dim3);
        size_t wts_base_offset = (i * wts_dim1 * wts_dim2 * wts_dim3) +
                                 (j * wts_dim2 * wts_dim3) + (k * wts_dim3);
        size_t zp_base_offset = (i * zp_dim1 * zp_dim2 * zp_dim3) +
                                (j * zp_dim2 * zp_dim3) + (k * zp_dim3);
        size_t scale_base_offset = (i * scale_dim1 * scale_dim2 * scale_dim3) +
                                   (j * scale_dim2 * scale_dim3) +
                                   (k * scale_dim3);

        // Copy data
        std::memcpy(&combined_data[base_offset], &wts_int16[wts_base_offset],
                    wts_dim3 * sizeof(uint16_t));
        std::memcpy(&combined_data[base_offset + wts_dim3],
                    &zp_int16[zp_base_offset], zp_dim3 * sizeof(uint16_t));
        std::memcpy(&combined_data[base_offset + wts_dim3 + zp_dim3],
                    &scale_int16[scale_base_offset],
                    scale_dim3 * sizeof(uint16_t));
      }
    }
  }

  return combined_data;
}

// Convert FP32 to INT16
static std::vector<uint16_t> np_fp32_2_int16(const std::vector<float> &input) {
  // Convert FP32 to uint32_t representation
  std::vector<uint32_t> uint32_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    uint32_input[i] = *reinterpret_cast<const uint32_t *>(&input[i]);
  }

  // Apply AIE SRS
  std::vector<int> aie_mode = {16, 16, 6, 1};
  std::vector<uint16_t> result = aie_srs(uint32_input, aie_mode);

  return result;
}

template <typename T>
std::vector<T> pad_inner_dim_to_multiple(const std::vector<T> &data,
                                         const std::vector<size_t> &shape,
                                         size_t multiple, T pad_value = T()) {
  size_t inner_dim_size = shape.back();
  size_t outer_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    outer_size *= shape[i];
  }
  size_t pad_size = (multiple - (inner_dim_size % multiple)) % multiple;
  if (pad_size == 0) {
    return data;
  }
  std::vector<size_t> new_shape = shape;
  new_shape.back() += pad_size;
  size_t new_inner_dim_size = inner_dim_size + pad_size;
  size_t total_size = outer_size * new_inner_dim_size;
  std::vector<T> padded_data(total_size, pad_value);
  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    for (size_t inner_idx = 0; inner_idx < inner_dim_size; ++inner_idx) {
      size_t original_index = outer_idx * inner_dim_size + inner_idx;
      size_t padded_index = outer_idx * new_inner_dim_size + inner_idx;
      padded_data[padded_index] = data[original_index];
    }
  }

  return padded_data;
}

std::vector<uint16_t>
concatenate_along_axis2(const std::vector<uint16_t> &bias_bf16,
                        const std::vector<size_t> &bias_dims,
                        const std::vector<uint16_t> &combined_data,
                        const std::vector<size_t> &combined_dims) {
  // Ensure input dimensions have 4D shapes
  if (bias_dims.size() != 4 || combined_dims.size() != 4) {
    throw std::invalid_argument("All input tensors must have 4 dimensions.");
  }

  // Verify that input sizes match their respective dimensions
  auto calculate_size = [](const std::vector<size_t> &dims) {
    return std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
  };

  size_t bias_size = calculate_size(bias_dims);
  size_t combined_size = calculate_size(combined_dims);

  if (bias_bf16.size() != bias_size) {
    throw std::invalid_argument(
        "Size of bias_bf16 does not match its dimensions. "
        "Expected: " +
        std::to_string(bias_size) +
        ", Got: " + std::to_string(bias_bf16.size()));
  }

  if (combined_data.size() != combined_size) {
    throw std::invalid_argument(
        "Size of combined_data does not match its dimensions. "
        "Expected: " +
        std::to_string(combined_size) +
        ", Got: " + std::to_string(combined_data.size()));
  }

  // Check dimensions for compatibility along axis 2
  if (bias_dims[0] != combined_dims[0] || bias_dims[1] != combined_dims[1] ||
      bias_dims[3] != combined_dims[3]) {
    throw std::invalid_argument(
        "Input dimensions are not compatible for concatenation along axis 2.");
  }

  // Calculate new dimensions after concatenation along axis 2
  std::vector<size_t> output_dims = {bias_dims[0], bias_dims[1],
                                     bias_dims[2] + combined_dims[2],
                                     bias_dims[3]};

  // Create output array
  std::vector<uint16_t> output_data(output_dims[0] * output_dims[1] *
                                    output_dims[2] * output_dims[3]);

  // Function to calculate the index based on the 4D dimensions
  auto get_index = [](size_t i, size_t j, size_t k, size_t l,
                      const std::vector<size_t> &dims) {
    return ((i * dims[1] + j) * dims[2] + k) * dims[3] + l;
  };

  // Fill the combined array
  for (size_t i = 0; i < output_dims[0]; ++i) {
    for (size_t j = 0; j < output_dims[1]; ++j) {
      for (size_t k = 0; k < output_dims[2]; ++k) {
        // Determine the source tensor and copy data
        size_t target_idx = get_index(i, j, k, 0, output_dims);

        if (k < bias_dims[2]) {
          // Copy data from bias_bf16
          size_t bias_idx = get_index(i, j, k, 0, bias_dims);
          std::memcpy(&output_data[target_idx], &bias_bf16[bias_idx],
                      bias_dims[3] * sizeof(uint16_t));
        } else {
          // Copy data from combined_data
          size_t combined_k = k - bias_dims[2];
          size_t combined_idx = get_index(i, j, combined_k, 0, combined_dims);
          std::memcpy(&output_data[target_idx], &combined_data[combined_idx],
                      combined_dims[3] * sizeof(uint16_t));
        }
      }
    }
  }

  return output_data;
}

std::vector<uint16_t> reshape_weights(int K_, int N_, int sv_K_num_, int sv_K_,
                                      int sv_N_num_, int sv_N_,
                                      int8_t *wts_input) {
  int dim0 = sv_K_num_;
  int dim1 = 2;
  int dim2 = sv_K_ / 8 / 2;
  int dim3 = 8;
  int dim4 = sv_N_num_;
  int dim5 = sv_N_ / 8;
  int dim6 = 8;

  std::vector<int8_t> wts(wts_input, wts_input + (K_ * N_));
  // for (int i = 0; i < wts.size(); i++) {
  //   int value = static_cast<int>(*(wts_input + i / 2));
  //   if (i & 1) {
  //     value = ((value & 0xF0) >> 4);
  //   } else {
  //     value = (value & 0x0F);
  //   }
  //   wts[i] = value;
  // }

  std::vector<int8_t> wts_reshape1(K_ * N_);
  for (int i0 = 0; i0 < dim0; i0++) {
    for (int i1 = 0; i1 < dim1; i1++) {
      for (int i2 = 0; i2 < dim2; i2++) {
        for (int i3 = 0; i3 < dim3; i3++) {
          for (int i4 = 0; i4 < dim4; i4++) {
            for (int i5 = 0; i5 < dim5; i5++) {
              for (int i6 = 0; i6 < dim6; i6++) {
                int src_index = i0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6 +
                                i1 * dim2 * dim3 * dim4 * dim5 * dim6 +
                                i2 * dim3 * dim4 * dim5 * dim6 +
                                i3 * dim4 * dim5 * dim6 + i4 * dim5 * dim6 +
                                i5 * dim6 + i6;
                int dst_index = i1 * dim0 * dim2 * dim3 * dim4 * dim5 * dim6 +
                                i4 * dim0 * dim2 * dim3 * dim5 * dim6 +
                                i0 * dim2 * dim3 * dim5 * dim6 +
                                i5 * dim2 * dim3 * dim6 + i2 * dim3 * dim6 +
                                i6 * dim3 + i3;

                wts_reshape1[dst_index] = wts[src_index];
              }
            }
          }
        }
      }
    }
  }

  int new_dim0 = dim1;
  int new_dim1 = dim4;
  int new_dim2 = dim0;
  std::vector<size_t> wts_shape1 = {
      static_cast<size_t>(new_dim0), static_cast<size_t>(new_dim1),
      static_cast<size_t>(new_dim2),
      static_cast<size_t>(K_ * N_ / new_dim0 / new_dim1 / new_dim2)};
  auto wts_int16 = compress_int4_to_int16(wts_reshape1, wts_shape1);

  return wts_int16;
}

std::vector<uint16_t>
reshape_and_compress_zp(int K_, int N_, int sv_K_num_, int sv_N_num_, int sv_N_,
                        int bfp16_grp_size_, int grp_per_sv, int8_t *zp_input) {
  int dim0 = sv_K_num_;
  int dim1 = 2;
  int dim2 = sv_N_num_;
  int dim3 = sv_N_;

  std::vector<int8_t> zp(zp_input, zp_input + (K_ / bfp16_grp_size_ * N_));
  // for (int i = 0; i < zp.size(); i++) {
  //   int value = static_cast<int>(*(zp_input + i / 2));
  //   // auto high_4bit = ((value & 0xF0) >> 4);
  //   // high_4bit = high_4bit > 7 ? high_4bit - 16 : high_4bit;
  //   // auto low_4bit = (value & 0x0F);
  //   // low_4bit = low_4bit > 7 ? low_4bit - 16 : low_4bit;
  //   if (i & 1) {
  //     value = ((value & 0xF0) >> 4);
  //   } else {
  //     value = (value & 0x0F);
  //   }
  //   zp[i] = value;
  // }
  std::vector<int8_t> rp_zp(K_ / bfp16_grp_size_ * grp_per_sv * N_);
  for (int i = 0; i < K_ / bfp16_grp_size_; i++) {
    for (int j = 0; j < grp_per_sv; j++) {
      int src_index = i * N_;
      int dst_index = (i * grp_per_sv + j) * N_;

      std::memcpy(rp_zp.data() + dst_index, zp.data() + src_index,
                  N_ * sizeof(int8_t));
    }
  }

  std::vector<int8_t> zp_reshape1(K_ / bfp16_grp_size_ * grp_per_sv * N_);

  for (int i0 = 0; i0 < dim0; i0++) {
    for (int i1 = 0; i1 < dim1; i1++) {
      for (int i2 = 0; i2 < dim2; i2++) {
        for (int i3 = 0; i3 < dim3; i3++) {
          int src_index =
              i0 * dim1 * dim2 * dim3 + i1 * dim2 * dim3 + i2 * dim3 + i3;
          int dst_index =
              i1 * dim0 * dim2 * dim3 + i2 * dim0 * dim3 + i0 * dim3 + i3;
          zp_reshape1[dst_index] = rp_zp[src_index];
        }
      }
    }
  }

  std::vector<size_t> zp_shape2 = {
      static_cast<size_t>(2), static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(sv_K_num_), static_cast<size_t>(sv_N_)};

  auto padded_zp =
      pad_inner_dim_to_multiple<int8_t>(zp_reshape1, zp_shape2, 128, 0);

  std::vector<size_t> zp_shape3 = {
      static_cast<size_t>(2), static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(sv_K_num_),
      static_cast<size_t>((sv_N_ + 127) / 128 * 128)};

  auto zp_int16 = compress_int4_to_int16(padded_zp, zp_shape3);

  return zp_int16; // Return the compressed version of zp
}

std::vector<uint16_t> pad_array(const std::vector<uint16_t> &input, int dim0,
                                int dim1, int dim2, int dim3, int pad_dim0,
                                int pad_dim3) {
  int padded_dim0 = dim0 + pad_dim0;
  int padded_dim3 = dim3 + pad_dim3;
  std::vector<uint16_t> output(padded_dim0 * dim1 * dim2 * padded_dim3, 0);
  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      for (int k = 0; k < dim2; ++k) {
        for (int l = 0; l < dim3; ++l) {
          int input_index =
              i * (dim1 * dim2 * dim3) + j * (dim2 * dim3) + k * dim3 + l;
          int output_index = i * (dim1 * dim2 * padded_dim3) +
                             j * (dim2 * padded_dim3) + k * padded_dim3 + l;
          output[output_index] = input[input_index];
        }
      }
    }
  }
  return output;
}

std::vector<uint16_t> reshape_and_compress_scale(int K_, int N_, int sv_K_num_,
                                                 int sv_N_num_, int sv_N_,
                                                 int bfp16_grp_size_,
                                                 int grp_per_sv,
                                                 float *scale_input) {

  int dim0 = sv_K_num_;
  int dim1 = 2;
  int dim2 = sv_N_num_;
  int dim3 = sv_N_;

  std::vector<float> scale(K_ / bfp16_grp_size_ * N_);
  for (int i = 0; i < scale.size(); i++) {
    scale[i] = *(scale_input + i);
  }
  std::vector<float> rp_scale(K_ / bfp16_grp_size_ * grp_per_sv * N_);
  for (int i = 0; i < K_ / bfp16_grp_size_; i++) {
    for (int j = 0; j < grp_per_sv; j++) {
      int src_index = i * N_;
      int dst_index = (i * grp_per_sv + j) * N_;

      std::memcpy(rp_scale.data() + dst_index, scale.data() + src_index,
                  N_ * sizeof(float));
    }
  }

  std::vector<float> scale_reshape1(K_ / bfp16_grp_size_ * grp_per_sv * N_);

  for (int i0 = 0; i0 < dim0; i0++) {
    for (int i1 = 0; i1 < dim1; i1++) {
      for (int i2 = 0; i2 < dim2; i2++) {
        for (int i3 = 0; i3 < dim3; i3++) {
          int src_index =
              i0 * dim1 * dim2 * dim3 + i1 * dim2 * dim3 + i2 * dim3 + i3;
          int dst_index =
              i1 * dim0 * dim2 * dim3 + i2 * dim0 * dim3 + i0 * dim3 + i3;

          scale_reshape1[dst_index] = rp_scale[src_index];
        }
      }
    }
  }

  auto scale_int16 = np_fp32_2_int16(scale_reshape1);

  return scale_int16; // Return the compressed scale
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::initialize_bfp16_wts(
    std::vector<uint8_t> &bo_map, int8_t *wts_input, int8_t *zp_input,
    float *bias_input, float *scale_input) {
  M_ = static_cast<int>(a_shape_[0]);

  //  std::cout << " M_ " << M_ << std::endl;
  //  std::cout << " K_ " << K_ << std::endl;
  //  std::cout << " N_ " << N_ << std::endl;
  //  std::cout << " bfp16_grp_size_ " << bfp16_grp_size_ << std::endl;

  auto grp_per_sv = bfp16_grp_size_ / (sv_K_ / 2);

  bo_map.resize(b_bo_size_, 0);
  auto sv_size_in_byte =
      (sv_N_ * sv_K_ / 2) / 2 + (sv_N_ / 2 + 63) / 64 * 64 + sv_N_ * 2;
  auto sv_size_in_int16 = sv_size_in_byte / 2;

  std::vector<size_t> wts_shape1 = {
      static_cast<size_t>(2), static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(sv_K_num_),
      static_cast<size_t>(sv_K_ / 2 * sv_N_ / 2 / 2)};
  auto wts_int16 =
      reshape_weights(K_, N_, sv_K_num_, sv_K_, sv_N_num_, sv_N_, wts_input);

  std::vector<size_t> zp_shape2 = {
      static_cast<size_t>(2), static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(sv_K_num_),
      static_cast<size_t>((sv_N_ / 2 + 63) / 64 * 64 / 2)};
  auto zp_int16 =
      reshape_and_compress_zp(K_, N_, sv_K_num_, sv_N_num_, sv_N_,
                              bfp16_grp_size_, grp_per_sv, zp_input);

  std::vector<size_t> scale_shape2 = {
      static_cast<size_t>(2),
      static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(sv_K_num_),
      static_cast<size_t>(sv_N_),
  };
  auto scale_int16 =
      reshape_and_compress_scale(K_, N_, sv_K_num_, sv_N_num_, sv_N_,
                                 bfp16_grp_size_, grp_per_sv, scale_input);

  std::vector<float> bias(N_);
  for (int i = 0; i < bias.size(); i++) {
    bias[i] = *(bias_input + i);
  }

  std::vector<uint16_t> bias_int16 = np_fp32_2_int16(bias);
  std::vector<size_t> bias_bf16_dims = {
      static_cast<size_t>(2), static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(1), static_cast<size_t>(sv_size_in_int16)};

  std::vector<size_t> combined_data_dims = {
      static_cast<size_t>(2), static_cast<size_t>(sv_N_num_),
      static_cast<size_t>(sv_K_num_), static_cast<size_t>(sv_size_in_int16)};

  int dim0 = 1;
  int dim1 = sv_N_num_;
  int dim2 = 1;
  int dim3 = sv_N_;
  int pad_dim0 = 1;
  int pad_dim3 = sv_size_in_int16 - dim3;

  std::vector<uint16_t> bias_bf16 =
      pad_array(bias_int16, dim0, dim1, dim2, dim3, pad_dim0, pad_dim3);

  auto combined_data = concatenate_along_axis3(
      wts_int16, wts_shape1[0], wts_shape1[1], wts_shape1[2], wts_shape1[3],
      zp_int16, zp_shape2[0], zp_shape2[1], zp_shape2[2], zp_shape2[3],
      scale_int16, scale_shape2[0], scale_shape2[1], scale_shape2[2],
      scale_shape2[3]);
  auto output_wts = concatenate_along_axis2(bias_bf16, bias_bf16_dims,
                                            combined_data, combined_data_dims);

  std::memcpy(bo_map.data(), output_wts.data(),
              output_wts.size() * sizeof(uint16_t));
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::tuple<mladf_matrix_shapes, std::vector<int64_t>, double>
mladfmatmulbias<InT, WtT, AccT, OutT>::map_padded_shape(int64_t M, int64_t K,
                                                        int64_t N) const {
  RYZENAI_LOG_TRACE("Map padded shape");
  DD_THROW_IF(M > KERNEL_M_MAX, ("M size exceeding ultimate shape limit"));
  std::set<int64_t> tile_m;
  mladf_matrix_shapes tiling_info(M, K, N);
  for (const auto &mat : supported_shapes_) {
    if (mat.K == K && mat.N == N && mat.M == M) {
      std::vector<int64_t> tiling_m;
      tiling_m.push_back(mat.M);
      if (m_tiling_cost_.find(mat.M) == m_tiling_cost_.end()) {
        return std::make_tuple(tiling_info, tiling_m, 100);
      }
      return std::make_tuple(tiling_info, tiling_m, m_tiling_cost_.at(mat.M));
    }
    if (mat.K == K && mat.N == N &&
        std::find(tile_m.begin(), tile_m.end(), mat.M) == tile_m.end()) {
      tile_m.insert(mat.M);
    }
  }
  // for efficiency purposes, remove M=1 for tiling for now
  if (tile_m.size() > 1 && tile_m.find(1) != tile_m.end()) {
    tile_m.erase(1);
  }
  std::pair<double, std::vector<int64_t>> tiling_info_m =
      minimum_tiles(tile_m, m_tiling_cost_, M);
  tiling_info.M =
      std::reduce(tiling_info_m.second.begin(), tiling_info_m.second.end());

  RYZENAI_LOG_TRACE("Tiling M shape: " + std::to_string(tiling_info.M));
  return std::make_tuple(tiling_info, tiling_info_m.second,
                         tiling_info_m.first);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::setup_instr_init() {}
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::setup_instr_registry(
    const std::map<std::string, std::any> &attr) {
  if (attr.find("shapes") != attr.end()) {
    RYZENAI_LOG_TRACE(
        "[WARNING] shapes attribute is set, the feature will be deprecated in "
        "future DD, supported search space is changed");
    thresholds_.clear();
    supported_shapes_.clear();
    all_thresholds_.clear();

    auto shapes = std::any_cast<std::vector<std::vector<int>>>(
        attr.find("shapes")->second);
    std::map<std::pair<int64_t, int64_t>, std::map<int, int, std::greater<int>>>
        m_shape_lists;
    for (auto iter : shapes) {
      std::pair<int64_t, int64_t> key = std::make_pair(iter[1], iter[2]);
      if (m_shape_lists.find(key) == m_shape_lists.end()) {
        m_shape_lists.insert({key, std::map<int, int, std::greater<int>>()});
      }
      m_shape_lists.at(key).emplace((int)iter[0], 1);
      supported_shapes_.push_back(
          mladf_matrix_shapes(iter[0], iter[1], iter[2], iter[3]));
    }
    for (const auto &[key, m_shape_list] : m_shape_lists) {
      if (m_shape_list.size() > 1) {
        int i = 0;
        for (auto iter = m_shape_list.begin(); iter != m_shape_list.end();
             iter++) {
          if (i == (m_shape_list.size() - 1)) {
            break;
          }
          int val = std::next(iter)->first;
          all_thresholds_[key].push_back(std::make_pair(val, iter->first));
          i++;
        }
      }
      auto iter = m_shape_list.end();
      iter--;
      all_thresholds_[key].push_back(std::make_pair(0, iter->first));
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::setup_supported_shapes() {
  Transaction &txn = Transaction::getInstance();
  constexpr int shape_M_idx = 5;
  constexpr int shape_K_idx = shape_M_idx + 1;
  constexpr int shape_N_idx = shape_M_idx + 2;
  constexpr int shape_Gs_idx = shape_M_idx + 3;
  std::map<std::pair<int64_t, int64_t>, std::map<int, int, std::greater<int>>>
      m_shape_lists;
  // auto start = std::chrono::steady_clock::now();
  const std::vector<std::string> &txn_file_names =
      txn.match_prefix("mladfmatmulbias_" + txn_fname_prefix_);
  for (const std::string &filename : txn_file_names) {
    std::stringstream filename_ss(filename);
    std::string token;
    uint8_t i = 0;
    mladf_matrix_shapes mat_shape;
    while (std::getline(filename_ss, token, '_')) {
      if (i >= shape_M_idx) {
        std::stringstream token_stream(token);
        if (i == shape_M_idx) {
          token_stream >> mat_shape.M;
        } else if (i == shape_K_idx) {
          token_stream >> mat_shape.K;
        } else if (i == shape_N_idx) {
          token_stream >> mat_shape.N;
        } else if (i == shape_Gs_idx) {
          token_stream >> mat_shape.Gs;
        }
      }
      i++;
    }
    supported_shapes_.push_back(mat_shape);
    std::pair<int64_t, int64_t> key = std::make_pair(mat_shape.K, mat_shape.N);
    if (m_shape_lists.find(key) == m_shape_lists.end()) {
      m_shape_lists.insert({key, std::map<int, int, std::greater<int>>()});
    }
    m_shape_lists.at(key).emplace((int)mat_shape.M, 1);
  }
  // auto end = std::chrono::steady_clock::now();
  // double elapsed = std::chrono::duration<double, std::milli>(end -
  // start).count(); std::cout << "matchning takes (ms) : " << elapsed << endl;
  for (const auto &[key, m_shape_list] : m_shape_lists) {
    if (m_shape_list.size() > 1) {
      int i = 0;
      for (auto iter = m_shape_list.begin(); iter != m_shape_list.end();
           iter++) {
        if (i == (m_shape_list.size() - 1)) {
          break;
        }
        int val = std::next(iter)->first;
        all_thresholds_[key].push_back(std::make_pair(val, iter->first));
        i++;
      }
    }
    auto iter = m_shape_list.end();
    iter--;
    all_thresholds_[key].push_back(std::make_pair(0, iter->first));
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
mladfmatmulbias<InT, WtT, AccT, OutT>::mladfmatmulbias(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  initialized_ = false;

  txnbin_a_header = {{"bfloat16", "a16f"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_b_header = {{"uint4", "w4"}, {"int4", "w3"}, {"bfp16ebs8", "w16bfp"}};
  txnbin_acc_header = {{"bfloat16", "acc16f"}};
  op_version_ = "v1";
  if (attr.find("op_version") != attr.end()) {
    if (attr.at("op_version").type() == typeid(std::vector<std::string>)) {
      const auto &attrs_vec = std::any_cast<const std::vector<std::string> &>(
          attr.at("op_version"));
      op_version_ = attrs_vec[0];
    } else if (attr.at("op_version").type() == typeid(std::string)) {
      op_version_ = std::any_cast<std::string>(attr.find("op_version")->second);
    }
    if (op_version_ != "v1" && op_version_ != "v2" && op_version_ != "flat") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }
  if (op_version_ == "v2") {
    is_bfp16_cast_gemm_ = true;
    txnbin_a_header = {{"bfloat16", "abfp16"}};
    // TODO: int4->w3 map remain to be used
    // txnbin_b_header = {{"uint4", "w4"}, {"int4", "w3"}};
    txnbin_b_header = {{"uint4", "wbfp16"}, {"int4", "wbfp16"}};
  }
  txn_fname_prefix_ =
      "mladf_2x4x4_" + op_version_ + "_" + txnbin_a_header.at(a_dtype_) +
      txnbin_b_header.at(b_dtype_) + txnbin_acc_header.at(c_dtype_);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("txn_fname_prefix : {}", txn_fname_prefix_));

  /* Set the appropriate kernel group id based on the DPU sequence execution
   * flow or transaction binary flow The transaction binary flow is enabled
   * only
   * for w4a16 and w3a16 GEMMs
   */
  if (a_dtype_ == "bfloat16") {
    a_dtype_size_ = sizeof(uint16_t);
  }
  if (b_dtype == "uint4") {
    sign = 0;
  } else {
    sign = 1;
  }

  mladfmatmulbias_id_ = mladfmatmulbias_count++;
  // made up numbers
  m_tiling_cost_ = {{1, 0.24},     {128, 1.0},    {256, 1.98},   {512, 3.94},
                    {1024, 7.8},   {2048, 15.5},  {384, 2.96},   {640, 4.92},
                    {768, 5.92},   {896, 6.91},   {1152, 8.7},   {1280, 9.77},
                    {1408, 10.77}, {1536, 11.73}, {1664, 12.73}, {1792, 13.71},
                    {1920, 14.71}};
  // std::call_once(supported_shapes_flag_, [this]() { setup_supported_shapes();
  // });
  setup_supported_shapes();
  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);
  // std::cout << "op_version_ " << op_version_ << std::endl;
  // std::cout << "XCLBIN_FNAME " << XCLBIN_FNAME << std::endl;
  // std::cout << "txn_fname_prefix_ " << txn_fname_prefix_ << std::endl;

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));

  if (load_xrt) {

    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
        XCLBIN_FNAME, 0, {},
        XclbinContainer::getInstance().get_xclbin_content(XCLBIN_FNAME));

    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_, [this]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this]() { setup_instr_init(); });
    }
    setup_instr_registry(attr);
  }

  // asscending sort
  std::sort(supported_shapes_.begin(), supported_shapes_.end(),
            [](const mladf_matrix_shapes &a, const mladf_matrix_shapes &b) {
              if (a.M != b.M) {
                return a.M < b.M;
              }
              if (a.K != b.K) {
                return a.K < b.K;
              }
              if (a.N != b.N) {
                return a.N < b.N;
              }
              return false;
            });

  // superkernel parameters not set through SHIM DDR
  params_bytes = 0;
  KERNEL_M_MAX = 4096;

  max_a_bo_size_ = 0;
  max_cast_ofm_bo_size_ = 0;
  max_cast_int4_ofm_bo_size_ = 0;

  max_c_bo_size_ = 0;
  max_m_ = 0;
  max_k_ = 0;
  max_n_ = 0;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header =
        "mladfmatmulbias_id M K N kernel_m kernel_k kernel_n Execute"
        "time(ns) num_aie_runs run_aie_time(ns) A_Pad_time(ns) "
        "C_Pad_time(ns) C_depad_time(ns) A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) CPU_accum_time(ns) "
        "Avg_time_per_aie_run(ns) group_size\n";
    RYZENAI_LOG_INFO(header);
  });
  RYZENAI_LOG_TRACE(
      "[MLADFMATMULBIAS] ID: " + std::to_string(mladfmatmulbias_id_) +
      ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
      a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_kernel_shapes_kn_mladf() const {
  if (a_dtype_ == "bfloat16") {
    bool bfound = false;
    auto i = 0u;
    for (i = 0; i < supported_shapes_.size(); i++) {
      // we assume for each M, the [K N] has same items.
      if (supported_shapes_[i].M != supported_shapes_[0].M) {
        break;
      }
      if ((w_shape_[0] <= supported_shapes_[i].K) &&
          (w_shape_[1] <= supported_shapes_[i].N)) {
        bfound = true;
        break;
      }
    }
    if (bfound) {
      kernel_x_shape_[1] = supported_shapes_[i].K;
      kernel_y_shape_[0] = supported_shapes_[i].K;
      kernel_y_shape_[1] = supported_shapes_[i].N;
      kernel_z_shape_[1] = supported_shapes_[i].N;
      thresholds_ = all_thresholds_.at(
          std::make_pair(supported_shapes_[i].K, supported_shapes_[i].N));
      return;
    } else {
      // cpu tiling of N is possible, but not gonna use it sunce this function
      // is called in fusion as well
      DD_THROW("Shape not supported currently, Hint: You may have changed "
               "search space by setting 'shapes' attribute");
    }
  } else {
    // Current support is only for Bf16 activation type
    throw std::runtime_error(
        "No Kernel exists for the current activation data type");
  }
}

// reformat const will take the constant tensors
// and put them into packed tensor format
// for model loading optimization, we have a flow where we want to
// pre-process the weights, in which case we dont write to xrt::bo
// this will be used as part of export_const_params flow where
// is_online = false
// const_vecs is a vector since this operator has some support for
// host tiling, so there can be more than 1 set of packed tensor
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::reformat_const(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr,
    std::vector<std::vector<std::uint8_t>> &const_vecs, bool is_online) {
  if (is_bfp16_cast_gemm_) {
    int8_t *weights = (int8_t *)const_params.at(0).data;
    int8_t *zeros = (int8_t *)const_params.at(3).data;
    float *scales = (float *)const_params.at(2).data;
    float *bias = (float *)const_params.at(1).data;

    std::tuple<size_t, size_t> w_shape = {const_params.at(0).shape.at(0),
                                          const_params.at(0).shape.at(1)};
    w_shape_[0] = std::get<0>(w_shape);
    w_shape_[1] = std::get<1>(w_shape);
    set_kernel_shapes_kn_mladf();

    int group_size = 32;
    std::string key = "group_size";
    if (attr.find(key) != attr.end()) {
      if (attr.at(key).type() == typeid(std::vector<int>)) {
        group_size = (std::any_cast<const std::vector<int> &>(attr.at(key)))[0];
      } else if (attr.at(key).type() == typeid(int)) {
        group_size = std::any_cast<int>(attr.find(key)->second);
      }
    }
    bfp16_grp_size_ = group_size;
    set_kernel_shapes_kn_mladf();
    kernel_x_shape_[0] = KERNEL_M_MAX;
    kernel_z_shape_[0] = KERNEL_M_MAX;
    w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
    w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);

    // wts(K,N), zp(K/GRP_SIZE,N), scale(K/GRP_SIZE,N), bias(1,N)
    // weights shape: (w_shape_[0], w_shape_[1]) -> (w_padded_shape_[0],
    // w_padded_shape_[1]) bias: w_shape_[1] -> w_padded_shape_[1] zeros:
    // (w_shape_[0]/group_size , w_shape_[1]) -> (w_padded_shape_[0]/group_size
    // , w_padded_shape_[1]) scales: (w_shape_[0]/group_size , w_shape_[1]) ->
    // (w_padded_shape_[0]/group_size , w_padded_shape_[1])
    std::vector<int8_t> pad_wts;
    std::vector<float> pad_bias;
    std::vector<float> pad_scales;
    std::vector<int8_t> pad_zeros;

    if (w_padded_shape_[1] > w_shape_[1] || w_padded_shape_[0] > w_shape_[0]) {
      pad_wts.resize(w_padded_shape_[0] * w_padded_shape_[1], 0);
      for (int i = 0; i < w_shape_[0]; i++) {
        for (int j = 0; j < w_shape_[1]; j++) {
          pad_wts[i * w_padded_shape_[1] + j] =
              *(weights + i * w_shape_[1] + j);
        }
      }
      weights = pad_wts.data();
      pad_bias.resize(w_padded_shape_[1], 0);
      for (int i = 0; i < w_shape_[1]; i++) {
        pad_bias[i] = *(bias + i);
      }
      bias = pad_bias.data();

      pad_scales.resize(w_padded_shape_[0] / group_size * w_padded_shape_[1],
                        0);
      for (int i = 0; i < w_shape_[0] / group_size; i++) {
        for (int j = 0; j < w_shape_[1]; j++) {
          pad_scales[i * w_padded_shape_[1] + j] =
              *(scales + i * w_shape_[1] + j);
        }
      }
      scales = pad_scales.data();

      pad_zeros.resize(w_padded_shape_[0] / group_size * w_padded_shape_[1], 0);
      for (int i = 0; i < w_shape_[0] / group_size; i++) {
        for (int j = 0; j < w_shape_[1]; j++) {
          pad_zeros[i * w_padded_shape_[1] + j] =
              *(zeros + i * w_shape_[1] + j);
        }
      }
      zeros = pad_zeros.data();
    }

    set_bfp16_kernel_shapes();
    xrt::bo b_bo;
    std::vector<uint8_t> bo_vec;

    WtT *bo_map = nullptr;
    auto b_format_start = GET_ELAPSED_TIME_NS();
    initialize_bfp16_wts(bo_vec, weights, zeros, bias, scales);

    if (is_online) {
      b_bo = xrt::bo(xrt_ctx_->get_context(), b_bo_size_,
                     xrt::bo::flags::host_only,
                     xrt_ctx_->get_kernel().group_id(0));
      bo_map = b_bo.map<WtT *>();
      std::memcpy(bo_map, bo_vec.data(), bo_vec.size());
    } else {
      bo_map = (WtT *)bo_vec.data();
    }
    auto b_format_stop = GET_ELAPSED_TIME_NS();
    b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);

    auto b_sync_start = GET_ELAPSED_TIME_NS();
    if (is_online) {
      b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto b_sync_stop = GET_ELAPSED_TIME_NS();
    b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

    if (is_online) {
      weights_bo_.push_back(b_bo);
    } else {
      const_vecs.push_back(bo_vec);
    }
  } else {
    // get original arguments from const Tensors
    int8_t *weights = (int8_t *)const_params.at(0).data;
    int8_t *zeros = (int8_t *)const_params.at(3).data;
    float *scales = (float *)const_params.at(2).data;
    float *bias = (float *)const_params.at(1).data;

    std::tuple<size_t, size_t> w_shape = {const_params.at(0).shape.at(0),
                                          const_params.at(0).shape.at(1)};

    int group_size = 128;
    std::string key = "group_size";
    if (attr.find(key) != attr.end()) {
      if (attr.at(key).type() == typeid(std::vector<int>)) {
        group_size = (std::any_cast<const std::vector<int> &>(attr.at(key)))[0];
      } else if (attr.at(key).type() == typeid(int)) {
        group_size = std::any_cast<int>(attr.find(key)->second);
      }
    }
    // Note: for mladf int8 gemm we had to change group id to 0
    const int group_id = 0;
    w_shape_[0] = std::get<0>(w_shape);
    w_shape_[1] = std::get<1>(w_shape);

    set_kernel_shapes_kn_mladf();

    // Use largest M dimension as the default. This has to correspond
    // to one of the available kernel sizes.
    //    NOTE: smaller M's can be selected in run_aie if needed
    kernel_x_shape_[0] = KERNEL_M_MAX;
    kernel_z_shape_[0] = KERNEL_M_MAX;

    /* Create weight BOs */
    // Create a BO for weight block and initialize to zero
    //    NOTE: We must initialize to zero here because the weight matrix
    //          shape might not be an integer multiple of the block size.
    //          Initializing the BOs to zero takes care of the padding
    //          without allocating any extra scratch space.

    // For int4 quantization the buffer also contains bias, zeros, and scales
    // the weights are tiled in zigzag w4 aligned subvolumes of 32x128 tiles
    // the first subvolume consists of bias that is padded with zeros
    // Rest of the subvolumes consist weights+scales+zeros in each tile
    // QuantMatrix class has helper functions to write the data into the
    // correct index

    w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
    w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);
    // The bfp16 kernel uses a block size of 4 for the default and 2 for the
    // updated overlay.
    int blk_size = (op_version_ == "v1") ? 2 : 4;

    mladfQuantMatrix<64, 32, 32, 32> buff_B1((int)kernel_y_shape_[0],
                                             (int)kernel_y_shape_[1], blk_size);
    mladfQuantMatrix<64, 128, 32, 128> buff_B2(
        (int)kernel_y_shape_[0], (int)kernel_y_shape_[1], blk_size);
    // iterate over kernel shaped blocks of the weight matrix
    for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
      for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
        auto b_format_start = GET_ELAPSED_TIME_NS();

        int block_size =
            (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;

        xrt::bo bo_;
        std::vector<std::uint8_t> bo_vec;

        WtT *bo_map = nullptr;

        if (is_online) {
          bo_ = xrt::bo(xrt_ctx_->get_context(), block_size,
                        xrt::bo::flags::host_only,
                        xrt_ctx_->get_kernel().group_id(group_id));
          bo_map = bo_.map<WtT *>();
          memset((void *)bo_map, 0, block_size);
        } else {
          bo_vec = std::vector<std::uint8_t>(block_size, 0);
          bo_map = (WtT *)bo_vec.data();
        }

        buff_B1.data = (mladfCoreSubv<32, 32, 32> *)bo_map;
        buff_B2.data = (mladfCoreSubv<128, 32, 128> *)bo_map;

        // first pack the bias (bf16)
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
          if (rb == 0) {
            (group_size < 128)
                ? buff_B1.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c])
                : buff_B2.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c]);
          }
        }
        // format quantized weights (int4/uint4)
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
               c += 2) {
            // NOTE: int8_t weights will be sign extended to int
            int x = weights[((rb + r) * w_shape_[1]) + (cb + c)];
            int y = weights[((rb + r) * w_shape_[1]) + (cb + c) + 1];
            if (b_dtype_ == "int4") {
              (group_size < 128)
                  ? buff_B1.quant(r, c) = ryzenai::pack_v2int4(x, y)
                  : buff_B2.quant(r, c) = ryzenai::pack_v2int4(x, y);
            } else {
              (group_size < 128)
                  ? buff_B1.quant(r, c) = ryzenai::pack_v2uint4(x, y)
                  : buff_B2.quant(r, c) = ryzenai::pack_v2uint4(x, y);
            }
          }
        }

        // Select the supported group_size
        if (group_size >= 128) {
          assert(group_size % 128 == 0 &&
                 "group_size should be div by 32 or 128");
          grp_size_ = 128;
        } else if (group_size >= 32) {
          assert(group_size % 32 == 0 &&
                 "group_size should be div by 32 or 128");
          grp_size_ = 32;
        }

        int repeat_count = group_size / grp_size_;
        // format the scales (bf16)
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
             r += group_size) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; c++) {
            for (int g = 0; g < repeat_count; g++) {
              (group_size < 128)
                  ? buff_B1.scale(r + g * grp_size_, c) =
                        ryzenai::float_to_bfloat16(scales[(
                            ((rb + r) * w_shape_[1] / group_size) + (cb + c))])
                  : buff_B2.scale(r + g * grp_size_, c) =
                        ryzenai::float_to_bfloat16(scales[(
                            ((rb + r) * w_shape_[1] / group_size) + (cb + c))]);
            }
          }
        }

        // format the zeros (int4)
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
             r += group_size) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
               c += 2) {
            size_t index = ((rb + r) * w_shape_[1] / (group_size)) + (cb + c);
            int x = zeros[index];
            int y = zeros[index + 1];
            int8_t pack_zeros;
            if (b_dtype_ == "int4") {
              pack_zeros = ryzenai::pack_v2int4(x, y);
            } else {
              pack_zeros = ryzenai::pack_v2uint4(x, y);
            }
            for (int g = 0; g < repeat_count; g++) {
              (group_size < 128)
                  ? buff_B1.zero(r + g * grp_size_, c) = pack_zeros
                  : buff_B2.zero(r + g * grp_size_, c) = pack_zeros;
            }
          }
        }
        auto b_format_stop = GET_ELAPSED_TIME_NS();
        b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);

        auto b_sync_start = GET_ELAPSED_TIME_NS();
        if (is_online) {
          bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        auto b_sync_stop = GET_ELAPSED_TIME_NS();
        b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

        if (is_online) {
          weights_bo_.push_back(bo_);
        } else {
          const_vecs.push_back(bo_vec);
        }
      }
    }
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
bool mladfmatmulbias<InT, WtT, AccT, OutT>::create_bo(void *usr_ptr,
                                                      size_t size,
                                                      int operand_index) {
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(usr_ptr);
  constexpr std::uint32_t MASK = ((1 << 12) - 1);
  if ((addr & MASK) != 0) {
    return false;
  }
  auto bo =
      xrt::bo(xrt_ctx_->get_context(), usr_ptr, size, xrt::bo::flags::host_only,
              xrt_ctx_->get_kernel().group_id(0));
  if (operand_index == 0) {
    a_bo_ = bo;
  } else if (operand_index == 1) {
    c_bo_ = bo;
  } else if (operand_index == 2) {
    cast_ofm_bo_ = bo;
  } else if (operand_index == 3) {
    cast_int4_ofm_bo_ = bo;
  } else {
    return false;
  }
  return true;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  std::vector<std::vector<std::uint8_t>> const_vecs;

  constexpr bool is_online = true;

  int num_preformat_tensors = 0;

  if (attr.find("num_preformat_tensors") != attr.end()) {
    num_preformat_tensors =
        std::any_cast<int>(attr.find("num_preformat_tensors")->second);
  }
  // tensor size is the max size for bo create, then share this single bo
  // between ops
  int share_tensor_size = 0;

  if (attr.find("tensor_size") != attr.end()) {
    share_tensor_size = std::any_cast<int>(attr.find("tensor_size")->second);
  }

  int use_host_buffer = 0;

  if (attr.find("use_host_buffer") != attr.end()) {
    use_host_buffer = std::any_cast<int>(attr.find("use_host_buffer")->second);
  }

  // Note: for mladf int8 gemm we had to change group id to 0
  const int group_id = 0;

  if (0 == num_preformat_tensors) {
    reformat_const(const_params, attr, const_vecs, is_online);
  } else {
    constexpr int PACKED_TENSOR_OFFSET = 1;
    for (int packed_tensor_idx = 0; packed_tensor_idx < num_preformat_tensors;
         packed_tensor_idx++) {
      const auto &const_tensor =
          const_params.at(packed_tensor_idx + PACKED_TENSOR_OFFSET);

      size_t const_tensor_size =
          std::accumulate(const_tensor.shape.begin(), const_tensor.shape.end(),
                          size_t{1}, std::multiplies{}) *
          Utils::get_size_of_type(const_tensor.dtype);

      std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(const_tensor.data);
      // Need 4K alignment
      constexpr std::uint32_t MASK = ((1 << 12) - 1);
      bool is_aligned = ((addr & MASK) == 0);
      size_t wts_tensor_size = const_tensor_size;
      if (use_host_buffer && (is_aligned)) {

        if (share_tensor_size) { // create bo with max size, share single bo and
                                 // load wts just-in-time
          weights_bo_.clear();
          wts_tensor_size =
              std::max((size_t)share_tensor_size, const_tensor_size);
        }
        xrt::bo const_bo = xrt::bo(xrt_ctx_->get_context(), const_tensor.data,
                                   wts_tensor_size, xrt::bo::flags::host_only,
                                   xrt_ctx_->get_kernel().group_id(group_id));
        const_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        weights_bo_.push_back(const_bo);
      } else {
        xrt::bo const_bo = xrt::bo(xrt_ctx_->get_context(), const_tensor_size,
                                   xrt::bo::flags::host_only,
                                   xrt_ctx_->get_kernel().group_id(group_id));

        constexpr int offset = 0;
        const_bo.write(const_tensor.data, const_tensor_size, offset);
        const_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weights_bo_.push_back(const_bo);
      }
    }

    // NOTE: Need to set these state parameters which are used
    //       for xrt::bo allocation that happens below
    std::tuple<size_t, size_t> w_shape = {const_params.at(0).shape.at(0),
                                          const_params.at(0).shape.at(1)};
    w_shape_[0] = std::get<0>(w_shape);
    w_shape_[1] = std::get<1>(w_shape);

    set_kernel_shapes_kn_mladf();
  }

  // a_bo, c_bo, a_bo_token, c_bo_token
  std::array<bool, 4> need_realloc = {false, false, false, false};

  if (attr.find("max_m") != attr.end()) {
    if (attr.at("max_m").type() == typeid(int)) {
      max_m_ = (size_t)std::any_cast<int>(attr.find("max_m")->second);
    } else if (attr.at("max_m").type() == typeid(std::vector<int>)) {
      max_m_ = (std::any_cast<const std::vector<int> &>(attr.at("max_m")))[0];
    }
    RYZENAI_LOG_TRACE("Updated max m to: " + std::to_string(max_m_));
  } else {
    max_m_ = KERNEL_M_MAX;
  }
  size_t a_bo_size = max_m_ * kernel_x_shape_[1];

  size_t c_bo_size = (kernel_x_shape_[1] <= 4096)
                         ? max_m_ * kernel_z_shape_[1]
                         : 2 * max_m_ * kernel_z_shape_[1];

  if (a_bo_size > max_a_bo_size_) {
    need_realloc.at(0) = true;
    max_a_bo_size_ = a_bo_size;
  }
  if (c_bo_size > max_c_bo_size_) {
    need_realloc.at(1) = true;
    max_c_bo_size_ = c_bo_size;
  }
  if (a_bo_size / max_m_ > max_k_) {
    need_realloc.at(2) = true;
    max_k_ = a_bo_size / max_m_;
  }
  if (c_bo_size / max_m_ > max_n_) {
    need_realloc.at(3) = true;
    max_n_ = c_bo_size / max_m_;
  }

  if ((std::count(need_realloc.begin(), need_realloc.end(), true) > 0)) {

    size_t A_BO_SIZE_TOKEN = max_k_ * a_dtype_size_;
    size_t C_BO_SIZE_TOKEN = max_n_ * sizeof(OutT);
    size_t A_BO_SIZE = max_a_bo_size_ * a_dtype_size_;
    size_t C_BO_SIZE = max_c_bo_size_ * sizeof(OutT);
    bool skip_input_creation = attr.find("skip_create_input") != attr.end();
    bool skip_output_creation = attr.find("skip_create_output") != attr.end();
    bool skip_token_creation = attr.find("skip_create_token") != attr.end();
    RYZENAI_LOG_TRACE("A_BO_SIZE: " + std::to_string(A_BO_SIZE));
    RYZENAI_LOG_TRACE("C_BO_SIZE: " + std::to_string(C_BO_SIZE));
    RYZENAI_LOG_TRACE("A_BO_SIZE_TOKEN: " + std::to_string(A_BO_SIZE_TOKEN));
    RYZENAI_LOG_TRACE("C_BO_SIZE_TOKEN: " + std::to_string(C_BO_SIZE_TOKEN));
    if (need_realloc.at(0) && !skip_input_creation) {
      RYZENAI_LOG_TRACE("REALLOCATING for a_bo_");
      a_bo_ =
          xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(group_id));
      if (is_bfp16_cast_gemm_) {
        cast_ofm_bo_size_ = 36 * (max_k_ / 8) * (max_m_ / 8) * 2;
        cast_int4_ofm_bo_size_ = 1152 * (max_k_ / 64 + 1) * (max_n_ / 32) * 2;
        if (cast_ofm_bo_size_ > max_cast_ofm_bo_size_) {
          max_cast_ofm_bo_size_ = cast_ofm_bo_size_;
          cast_ofm_bo_size_ = {};
          cast_ofm_bo_ = xrt::bo(xrt_ctx_->get_context(), max_cast_ofm_bo_size_,
                                 xrt::bo::flags::host_only,
                                 xrt_ctx_->get_kernel().group_id(0));
        }
        if (cast_int4_ofm_bo_size_ > max_cast_int4_ofm_bo_size_) {
          max_cast_int4_ofm_bo_size_ = cast_int4_ofm_bo_size_;
          cast_int4_ofm_bo_ = {};
          cast_int4_ofm_bo_ = xrt::bo(
              xrt_ctx_->get_context(), max_cast_int4_ofm_bo_size_,
              xrt::bo::flags::host_only, xrt_ctx_->get_kernel().group_id(0));
        }
      }
    }

    if (need_realloc.at(1) && !skip_output_creation) {
      RYZENAI_LOG_TRACE("REALLOCATING for c_bo_");
      c_bo_ =
          xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE, xrt::bo::flags::host_only,
                  xrt_ctx_->get_kernel().group_id(group_id));
    }
    if (need_realloc.at(2) && !skip_token_creation) {
      RYZENAI_LOG_TRACE("REALLOCATING for a_bo_token_");
      a_bo_token_ = xrt::bo(xrt_ctx_->get_context(), A_BO_SIZE_TOKEN,
                            xrt::bo::flags::host_only,
                            xrt_ctx_->get_kernel().group_id(group_id));
    }
    if (need_realloc.at(3) && !skip_token_creation) {
      RYZENAI_LOG_TRACE("REALLOCATING for c_bo_token_");
      c_bo_token_ = xrt::bo(xrt_ctx_->get_context(), C_BO_SIZE_TOKEN,
                            xrt::bo::flags::host_only,
                            xrt_ctx_->get_kernel().group_id(group_id));
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<std::vector<std::uint8_t>>
mladfmatmulbias<InT, WtT, AccT, OutT>::export_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  std::vector<std::vector<std::uint8_t>> const_vecs;

  constexpr bool is_online = false;

  reformat_const(const_params, attr, const_vecs, is_online);

  return const_vecs;
}

// specialization for ML-ADF, invoked in set_kernel_shapes_m if is_mladf_enabled
// is TRUE
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_kernel_shapes_m_mladf(
    int64_t input_m) {
  if (a_dtype_ == "bfloat16") {
    if (thresholds_.size()) {
      bool found = false;
      for (const auto &threshold : thresholds_) {
        if ((size_t)input_m > threshold.first) {
          kernel_x_rows = threshold.second;
          found = true;
          break;
        }
      }
      if (!found) {
        DD_THROW("M not supported in current suppoted shapes");
      }
    } else {
      DD_THROW("Available M sizes is 0 can't set M shape");
    }
  } else {
    throw std::runtime_error(
        "No Kernel exists for the chosen activation shape and data type");
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_bfp16_kernel_shapes() {
  // K_ = static_cast<int>(w_shape_[0]);
  // N_ = static_cast<int>(w_shape_[1]);
  K_ = static_cast<int>(w_padded_shape_[0]);
  N_ = static_cast<int>(w_padded_shape_[1]);
  sv_K_ = 64;
  sv_N_ = 64;
  // if (K_ == 4096) {
  //   sv_K_ = 64;
  //   sv_N_ = 64;
  // } else if (K_ == 13696 && N_ == 4096) {
  //   sv_K_ = 64;
  //   sv_N_ = 64;
  // } else if (K_ == 11008 && N_ == 4096) {
  //   sv_K_ = 64;
  //   sv_N_ = 128;
  // } else {
  //   throw std::runtime_error("bfp16_gemm K, N unsupport " +
  //   std::to_string(K_) +
  //                            " " + std::to_string(N_));
  // }
  sv_K_num_ = K_ / sv_K_;
  sv_N_num_ = N_ / sv_N_;
  sv_size_in_byte_ =
      (sv_N_ * sv_K_ / 2) / 2 + (sv_N_ / 2 + 63) / 64 * 64 + sv_N_ * 2;

  b_bo_size_ = sv_size_in_byte_ * 2 * (sv_K_num_ + 1) * (sv_N_num_);
  //  std::cout << "K_ " << K_ << std::endl;
  //  std::cout << "N_ " << N_ << std::endl;
  //  std::cout << "sv_K_ " << sv_K_ << std::endl;
  //  std::cout << "sv_N_ " << sv_N_ << std::endl;
  //  std::cout << "sv_K_num_ " << sv_K_num_ << std::endl;
  //  std::cout << "sv_N_num_ " << sv_N_num_ << std::endl;
  //  std::cout << "b_bo_size_ " << b_bo_size_ << std::endl;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::run_aie(InT *a, xrt::bo &w_bo,
                                                    int64_t *input_shape,
                                                    bool wait) {
  // NOTE: Here we select the DPU sequence to use based on the
  //       number of rows in the input. This allows us to optimize
  //       kernels for both prefill and token generation phases
  //       of LLM inference. All kernels share the same weight
  //       buffer. The input buffer is allocated to be big enough
  //       for the largest kernel.
  //

  auto a_bo_run_aie = a_bo_;
  auto c_bo_run_aie = c_bo_;
  if (input_shape[0] == 1) {
    a_bo_run_aie = a_bo_token_;
    c_bo_run_aie = c_bo_token_;
  }

  auto [tiling_info, tiling_info_m, cost] =
      map_padded_shape(input_shape[0], kernel_x_shape_[1], kernel_y_shape_[1]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(tiling_info.M) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);

  if (tiling_info_m.size() > 1) {
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = generate_fused_txnbin(tiling_info, tiling_info_m,
                                                 kernel_x_shape_[1],
                                                 kernel_y_shape_[1], grp_size_);
        auto instr = std::make_pair(instr_bo_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }

  // Pad and copy input activation to host BO memory
  // NOTE: BOs are allocated in the constructor to
  //       support the largest kernel size, so all of these
  //       memory accesses will be within bounds

  /*
   * Format A matrix for BF16 kernel
   * Since bf16 is not a natively supported dtype we
   * use int16_t buffers for activations and populate them
   */
  auto a_copy_start = GET_ELAPSED_TIME_NS();

  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);

  uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
  auto input_size = kernel_x_rows * kernel_x_shape_[1] * a_dtype_size_;
  memset((void *)a_map, 0, input_size);

  uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);

  auto a_sz = a_shape_[0] * a_shape_[1];

  for (int i = 0; i < input_shape[0]; ++i) {
    // copy row from the source tile
    memcpy((void *)&a_map[i * kernel_x_shape_[1]], (void *)&a[i * a_shape_[1]],
           input_shape[1] * a_dtype_size_);
  }

  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  // sync input activation to device memory
  auto a_sync_start = GET_ELAPSED_TIME_NS();

  a_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0);

  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the GEMM kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();

  // kernel call for GEMM that supports transaction binary flow

  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, a_bo_run_aie, w_bo, c_bo_run_aie,
      c_bo_run_aie, 0, wait, false);
  if (wait) {
    num_run_aie_++;

    auto run_aie_stop = GET_ELAPSED_TIME_NS();
    // sync output activation to host memory
    auto c_sync_start = GET_ELAPSED_TIME_NS();
    c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                      kernel_x_rows * kernel_z_shape_[1] * sizeof(OutT), 0);
    auto c_sync_stop = GET_ELAPSED_TIME_NS();

    a_copy_time_ += static_cast<int64_t>(a_copy_stop - a_copy_start);
    a_sync_time_ += static_cast<int64_t>(a_sync_stop - a_sync_start);
    c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);
    run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::run_aie_2(InT *a, xrt::bo &w_bo,
                                                      int64_t *input_shape,
                                                      bool wait) {
  // NOTE: Here we select the DPU sequence to use based on the
  //       number of rows in the input. This allows us to optimize
  //       kernels for both prefill and token generation phases
  //       of LLM inference. All kernels share the same weight
  //       buffer. The input buffer is allocated to be big enough
  //       for the largest kernel.
  //

  auto a_bo_run_aie = a_bo_;
  auto c_bo_run_aie = c_bo_;
  if (input_shape[0] == 1) {
    a_bo_run_aie = a_bo_token_;
    c_bo_run_aie = c_bo_token_;
  }

  set_kernel_shapes_m_mladf(input_shape[0]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(kernel_x_rows) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);

  RYZENAI_LOG_TRACE("instr_bo_key = " + instr_bo_key);

  // Pad and copy input activation to host BO memory
  // NOTE: BOs are allocated in the constructor to
  //       support the largest kernel size, so all of these
  //       memory accesses will be within bounds

  /*
   * Format A matrix for BF16 kernel
   * Since bf16 is not a natively supported dtype we
   * use int16_t buffers for activations and populate them
   */
  auto a_copy_start = GET_ELAPSED_TIME_NS();

  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);

  uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();
  auto input_size = kernel_x_rows * kernel_x_shape_[1] * a_dtype_size_;
  memset((void *)a_map, 0, input_size);

  uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);

  auto a_sz = a_shape_[0] * a_shape_[1];

  for (int i = 0; i < input_shape[0]; ++i) {
    // copy row from the source tile
    memcpy((void *)&a_map[i * kernel_x_shape_[1]], (void *)&a[i * a_shape_[1]],
           input_shape[1] * a_dtype_size_);
  }

  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  // sync input activation to device memory
  auto a_sync_start = GET_ELAPSED_TIME_NS();

  a_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0);

  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the GEMM kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();

  // kernel call for GEMM that supports transaction binary flow

  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, a_bo_run_aie, w_bo, c_bo_run_aie,
      c_bo_run_aie, 0, wait, false);
  if (wait) {
    num_run_aie_++;

    auto run_aie_stop = GET_ELAPSED_TIME_NS();
    // sync output activation to host memory
    auto c_sync_start = GET_ELAPSED_TIME_NS();
    c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                      kernel_x_rows * kernel_z_shape_[1] * sizeof(OutT), 0);
    auto c_sync_stop = GET_ELAPSED_TIME_NS();

    a_copy_time_ += static_cast<int64_t>(a_copy_stop - a_copy_start);
    a_sync_time_ += static_cast<int64_t>(a_sync_stop - a_sync_start);
    c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);
    run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_shape(
    std::vector<size_t> a_shape, std::vector<size_t> wt_shape, int group_size) {
  a_shape_[0] = a_shape[0];
  a_shape_[1] = a_shape[1];
  c_shape_[0] = a_shape[0];
  w_shape_[0] = wt_shape.at(0);
  w_shape_[1] = wt_shape.at(1);
  set_kernel_shapes_kn_mladf();
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;
  // kernel_z_shape_[1] = wt_shape.at(1);
  c_shape_[1] = wt_shape[1];
  if (is_bfp16_cast_gemm_) {
    grp_size_ = 0;
  } else {
    grp_size_ = group_size;
  }
  auto [tiling_info, tiling_info_m, cost] =
      map_padded_shape(a_shape_[0], kernel_x_shape_[1], kernel_y_shape_[1]);
  RYZENAI_LOG_TRACE("Map padded shape: M: " + std::to_string(a_shape_[0]) +
                    " mapped to: ");
  for (const auto &tile : tiling_info_m) {
    RYZENAI_LOG_TRACE("Tile: " + std::to_string(tile));
  }
  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(tiling_info.M) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  if (tiling_info_m.size() > 1) {
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = generate_fused_txnbin(tiling_info, tiling_info_m,
                                                 kernel_x_shape_[1],
                                                 kernel_y_shape_[1], grp_size_);
        auto instr = std::make_pair(instr_bo_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::set_shape_2(
    std::vector<size_t> a_shape, std::vector<size_t> wt_shape, int group_size) {
  a_shape_[0] = a_shape[0];
  a_shape_[1] = a_shape[1];
  c_shape_[0] = a_shape[0];
  w_shape_[0] = wt_shape.at(0);
  w_shape_[1] = wt_shape.at(1);
  set_kernel_shapes_kn_mladf();
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;
  // kernel_z_shape_[1] = wt_shape.at(1);
  c_shape_[1] = wt_shape[1];
  grp_size_ = group_size;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute_2(
    std::vector<xrt::bo> &inputs, std::vector<xrt::bo> &outputs, bool wait) {

  int64_t input_shape[2];
  input_shape[0] = std::min(a_shape_[0], kernel_x_shape_[0]);
  input_shape[1] = std::min(a_shape_[1], kernel_x_shape_[1]);
  set_kernel_shapes_m_mladf(input_shape[0]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(kernel_x_rows) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);

  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  auto kernel_ = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, inputs[0], inputs[1], outputs[0],
      outputs[0], 0, wait, false);
  if (wait) {

    if (c_shape_[1] < kernel_z_shape_[1]) {
      outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto c_map = outputs[0].map<AccT *>();
      int64_t output_shape[2];
      output_shape[0] = std::min(c_shape_[0], kernel_z_shape_[0]);
      output_shape[1] = std::min(c_shape_[1], kernel_z_shape_[1]);

      for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c_map[i * c_shape_[1]],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(AccT));
      }
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute(
    std::vector<xrt::bo> &inputs, std::vector<xrt::bo> &outputs, bool wait) {
  int64_t input_shape[2];
  input_shape[0] = std::min(a_shape_[0], kernel_x_shape_[0]);
  input_shape[1] = std::min(a_shape_[1], kernel_x_shape_[1]);
  auto [tiling_info, tiling_info_m, cost] =
      map_padded_shape(input_shape[0], kernel_x_shape_[1], kernel_y_shape_[1]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(tiling_info.M) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  if (is_bfp16_cast_gemm_) {
    instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                   std::to_string(tiling_info.M) + "_" +
                   std::to_string(kernel_x_shape_[1]) + "_" +
                   std::to_string(kernel_y_shape_[1]);
  }
  if (tiling_info_m.size() > 1) {
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = generate_fused_txnbin(tiling_info, tiling_info_m,
                                                 kernel_x_shape_[1],
                                                 kernel_y_shape_[1], grp_size_);
        auto instr = std::make_pair(instr_bo_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  if (is_bfp16_cast_gemm_) {
    // std::cout << "intput size: " << inputs[0].size() << " cast_ofm: " <<
    // cast_ofm_bo_.size() << "  wts size "
    //   << inputs[1].size() << " cast_int4: " << cast_int4_ofm_bo_.size() << "
    //   output: " << outputs[0].size() << " " << std::endl;
    // std::cout << cast_ofm_bo_.address() << " " << cast_int4_ofm_bo_.address()
    // << std::endl;
    auto run = kernel_(
        2, instr_bo, instr_bo_words, inputs[0].address() + DDR_AIE_ADDR_OFFSET,
        inputs[1].address() + DDR_AIE_ADDR_OFFSET,
        outputs[0].address() + DDR_AIE_ADDR_OFFSET,
        cast_ofm_bo_.address() + DDR_AIE_ADDR_OFFSET,
        (!cast_int4_ofm_bo_) // in bfp16 kernel, if M<512, this bo is useless
            ? 0
            : cast_int4_ofm_bo_.address() + DDR_AIE_ADDR_OFFSET);
    if (wait) {
      run.wait2();
    }
  } else {
    ryzenai::dynamic_dispatch::execute_kernel(
        kernel_, 2, instr_bo, instr_bo_words, inputs[0], inputs[1], outputs[0],
        outputs[0], 0, wait, false);
  }
  if (wait) {

    if (c_shape_[1] < kernel_z_shape_[1]) {
      outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto c_map = outputs[0].map<AccT *>();
      int64_t output_shape[2];
      output_shape[0] = std::min(c_shape_[0], kernel_z_shape_[0]);
      output_shape[1] = std::min(c_shape_[1], kernel_z_shape_[1]);

      for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c_map[i * c_shape_[1]],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(AccT));
      }
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute_2(
    std::vector<uint64_t> &inputs, std::vector<xrt::bo> &outputs, bool wait) {
  int64_t input_shape[2];
  input_shape[0] = std::min(a_shape_[0], kernel_x_shape_[0]);
  input_shape[1] = std::min(a_shape_[1], kernel_x_shape_[1]);

  set_kernel_shapes_m_mladf(input_shape[0]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(kernel_x_rows) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);

  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  auto kernel_ = xrt_ctx_->get_kernel();

  //  kernel call for GEMM that supports transaction binary flow

  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, inputs[0], inputs[1], outputs[0],
      outputs[0], 0, wait, false);
  if (wait) {
    if (c_shape_[1] < kernel_z_shape_[1]) {

      outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto c_map = outputs[0].map<AccT *>();
      int64_t output_shape[2];
      output_shape[0] = std::min(c_shape_[0], kernel_z_shape_[0]);
      output_shape[1] = std::min(c_shape_[1], kernel_z_shape_[1]);

      for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c_map[i * c_shape_[1]],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(AccT));
      }
    }
  }
}

static std::vector<uint32_t> read_file(const std::string &filePath) {
  std::ifstream fileStream(filePath);
  if (!fileStream.is_open()) {
    std::cerr << "Failed to open file " << filePath << "!" << std::endl;
    throw std::runtime_error("Failed to open file " + filePath + "!");
    return {};
  }

  std::vector<uint32_t> buffer;
  uint32_t temp;

  while (fileStream >> std::hex >> temp) {
    buffer.push_back(temp);
  }

  fileStream.close();
  return buffer;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute(
    std::vector<uint64_t> &inputs, std::vector<xrt::bo> &outputs, bool wait) {
  int64_t input_shape[2];
  input_shape[0] = std::min(a_shape_[0], kernel_x_shape_[0]);
  input_shape[1] = std::min(a_shape_[1], kernel_x_shape_[1]);

  auto [tiling_info, tiling_info_m, cost] =
      map_padded_shape(input_shape[0], kernel_x_shape_[1], kernel_y_shape_[1]);

  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(tiling_info.M) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  if (tiling_info_m.size() > 1) {
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = generate_fused_txnbin(tiling_info, tiling_info_m,
                                                 kernel_x_shape_[1],
                                                 kernel_y_shape_[1], grp_size_);
        auto instr = std::make_pair(instr_bo_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }

  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  auto kernel_ = xrt_ctx_->get_kernel();

  //  kernel call for GEMM that supports transaction binary flow
  if (is_bfp16_cast_gemm_) {
    ryzenai::dynamic_dispatch::execute_kernel(
        kernel_, 2, instr_bo, instr_bo_words, inputs[0], cast_ofm_bo_,
        inputs[1], cast_int4_ofm_bo_, outputs[0], wait, false);
  } else {
    ryzenai::dynamic_dispatch::execute_kernel(
        kernel_, 2, instr_bo, instr_bo_words, inputs[0], inputs[1], outputs[0],
        outputs[0], 0, wait, false);
  }

  if (wait) {
    if (c_shape_[1] < kernel_z_shape_[1]) {

      outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto c_map = outputs[0].map<AccT *>();
      int64_t output_shape[2];
      output_shape[0] = std::min(c_shape_[0], kernel_z_shape_[0]);
      output_shape[1] = std::min(c_shape_[1], kernel_z_shape_[1]);

      for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c_map[i * c_shape_[1]],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(AccT));
      }
    }
  }
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<xrt::bo> mladfmatmulbias<InT, WtT, AccT, OutT>::get_inputs(int M) {
  if (M == 1) {
    return {a_bo_token_};
  } else {
    return {a_bo_};
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<xrt::bo> mladfmatmulbias<InT, WtT, AccT, OutT>::get_const() {
  return weights_bo_;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<xrt::bo> mladfmatmulbias<InT, WtT, AccT, OutT>::get_outputs(int M) {
  if (M == 1) {
    return {c_bo_token_};
  } else {
    return {c_bo_};
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute_2(
    //    InT *a, const std::tuple<int, int> &a_shape, OutT *c) {
    std::vector<Tensor> &input_Tensor, std::vector<Tensor> &output_Tensor) {
  execute_internal_2(input_Tensor, output_Tensor, -1);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute(
    //    InT *a, const std::tuple<int, int> &a_shape, OutT *c) {
    std::vector<Tensor> &input_Tensor, std::vector<Tensor> &output_Tensor) {
  execute_internal(input_Tensor, output_Tensor, -1);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute_internal_2(
    //    InT *a, const std::tuple<int, int> &a_shape, OutT *c) {
    std::vector<Tensor> &input_Tensor, std::vector<Tensor> &output_Tensor,
    int wts_index, bool wait) {
  // get original arguments from input/output Tensor
  InT *a = (InT *)input_Tensor.at(0).data;
  std::tuple<size_t, size_t> a_shape = {input_Tensor.at(0).shape.at(0),
                                        input_Tensor.at(0).shape.at(1)};
  OutT *c = (OutT *)output_Tensor.at(0).data;
  auto exec_start = GET_ELAPSED_TIME_NS();
  a_sync_time_ = 0;
  c_sync_time_ = 0;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  a_shape_[0] = std::get<0>(a_shape);
  a_shape_[1] = std::get<1>(a_shape);
  c_shape_[0] = std::get<0>(a_shape);
  c_shape_[1] = w_shape_[1];

  AccT *c_acc;
  if constexpr (std::is_same_v<AccT, OutT>) {
    c_acc = reinterpret_cast<AccT *>(c);
  } else {
    if (c_acc_vec_.size() != (c_shape_[0] * c_shape_[1])) {
      c_acc_vec_.resize(c_shape_[0] * c_shape_[1]);
    }
    c_acc = c_acc_vec_.data();
  }
  for (int64_t ra = 0; ra < a_shape_[0]; ra += kernel_x_shape_[0]) {
    for (int64_t cb = 0; cb < w_shape_[1]; cb += kernel_y_shape_[1]) {
      int k = 0;
      // compute row major tile index for weight BOs
      int64_t tile_pitch = w_padded_shape_[1] / kernel_y_shape_[1];
      int64_t tile_row = k / kernel_y_shape_[0];
      int64_t tile_col = cb / kernel_y_shape_[1];
      int64_t tile_idx = tile_row * tile_pitch + tile_col;
      if (wts_index >= 0) {
        tile_idx = wts_index;
      }
      // compute shape of current input tile
      int64_t input_shape[2];
      input_shape[0] = std::min(a_shape_[0] - ra, kernel_x_shape_[0]);
      input_shape[1] = std::min(a_shape_[1] - k, kernel_x_shape_[1]);

      run_aie_2(&a[ra * a_shape_[1] + k], weights_bo_[tile_idx], input_shape);

      // compute shape of current output tile
      int64_t output_shape[2];
      output_shape[0] = std::min(c_shape_[0] - ra, kernel_z_shape_[0]);
      output_shape[1] = std::min(c_shape_[1] - cb, kernel_z_shape_[1]);

      auto c_copy_start = GET_ELAPSED_TIME_NS();
      // initialize the output tile

      auto c_map = c_bo_.map<AccT *>();
      if (output_shape[0] == 1) {
        c_map = c_bo_token_.map<AccT *>();
      } else {
        c_map = c_bo_.map<AccT *>();
      }
      for (int i = 0; i < output_shape[0]; ++i) {
        memcpy((void *)&c_acc[(ra + i) * c_shape_[1] + cb],
               (void *)&c_map[i * kernel_z_shape_[1]],
               output_shape[1] * sizeof(AccT));
      }

      auto c_copy_stop = GET_ELAPSED_TIME_NS();
      c_copy_time_ += static_cast<int64_t>(c_copy_stop - c_copy_start);
    }
  }

  auto exec_end = GET_ELAPSED_TIME_NS();

  int64_t a_pad_time = 0;
  int64_t c_pad_time = 0;
  int64_t cpu_depad_time = 0;

  if (debug_) {
    // Write input / output matrix to file.
    std::string a_fname = "ryzenai_qlinear2_" +
                          std::to_string(mladfmatmulbias_id_) + "_" +
                          std::to_string(num_execute_) + "_a.txt";
    std::string c_fname = "ryzenai_qlinear2_" +
                          std::to_string(mladfmatmulbias_id_) + "_" +
                          std::to_string(num_execute_) + "_c.txt";

    Utils::write_buffer_to_file(a, a_shape_[0] * a_shape_[1], a_fname);
    Utils::write_buffer_to_file(c, c_shape_[0] * c_shape_[1], c_fname);
  }
  num_execute_++;

  RYZENAI_LOG_INFO(
      std::to_string(mladfmatmulbias_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_pad_time) + " " + std::to_string(c_pad_time) + " " +
      std::to_string(cpu_depad_time) + " " + std::to_string(a_copy_time_) +
      " " + std::to_string(a_sync_time_) + " " + std::to_string(c_copy_time_) +
      " " + std::to_string(c_sync_time_) + " " + std::to_string(cpu_acc_time_) +
      " " + std::to_string((double)run_aie_time_ / num_run_aie_) + " " +
      std::to_string(grp_size_) + "\n");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::execute_internal(
    //    InT *a, const std::tuple<int, int> &a_shape, OutT *c) {
    std::vector<Tensor> &input_Tensor, std::vector<Tensor> &output_Tensor,
    int wts_index, bool wait) {
  // get original arguments from input/output Tensor
  InT *a = (InT *)input_Tensor.at(0).data;
  std::tuple<size_t, size_t> a_shape = {input_Tensor.at(0).shape.at(0),
                                        input_Tensor.at(0).shape.at(1)};
  OutT *c = (OutT *)output_Tensor.at(0).data;
  auto exec_start = GET_ELAPSED_TIME_NS();
  a_sync_time_ = 0;
  c_sync_time_ = 0;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  a_shape_[0] = std::get<0>(a_shape);
  a_shape_[1] = std::get<1>(a_shape);
  c_shape_[0] = std::get<0>(a_shape);
  c_shape_[1] = w_shape_[1];

  AccT *c_acc;
  if constexpr (std::is_same_v<AccT, OutT>) {
    c_acc = reinterpret_cast<AccT *>(c);
  } else {
    if (c_acc_vec_.size() != (c_shape_[0] * c_shape_[1])) {
      c_acc_vec_.resize(c_shape_[0] * c_shape_[1]);
    }
    c_acc = c_acc_vec_.data();
  }
  auto [tiling_info, tiling_info_m, cost] = map_padded_shape(
      input_Tensor.at(0).shape.at(0), kernel_x_shape_[1], kernel_y_shape_[1]);
  std::string instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                             std::to_string(tiling_info.M) + "_" +
                             std::to_string(kernel_x_shape_[1]) + "_" +
                             std::to_string(kernel_y_shape_[1]) + "_" +
                             std::to_string(grp_size_);
  if (is_bfp16_cast_gemm_) {
    instr_bo_key = "mladfmatmulbias_" + txn_fname_prefix_ + "_" +
                   std::to_string(tiling_info.M) + "_" +
                   std::to_string(kernel_x_shape_[1]) + "_" +
                   std::to_string(kernel_y_shape_[1]);
  }
  RYZENAI_LOG_TRACE("instr_bo_key = " + instr_bo_key);

  if (tiling_info_m.size() > 1) {
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = generate_fused_txnbin(tiling_info, tiling_info_m,
                                                 kernel_x_shape_[1],
                                                 kernel_y_shape_[1], grp_size_);
        auto instr = std::make_pair(instr_bo_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }
  // std::cout << instr_bo_key << std::endl;
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  // move a's data
  auto a_bo_run_aie = a_bo_;
  auto c_bo_run_aie = c_bo_;
  if (a_shape_[0] == 1) {
    a_bo_run_aie = a_bo_token_;
    c_bo_run_aie = c_bo_token_;
  }
  // copy input data to a bo
  auto input_size = a_shape_[0] * kernel_x_shape_[1] * a_dtype_size_;
  int64_t input_1_size = std::min(a_shape_[1], kernel_x_shape_[1]);
  auto a_copy_start = GET_ELAPSED_TIME_NS();

  uint16_t *a_map = a_bo_run_aie.map<uint16_t *>();

  memset((void *)a_map, 0, input_size);

  uint16_t *a_u16 = reinterpret_cast<uint16_t *>(a);

  for (int i = 0; i < a_shape_[0]; ++i) {
    // copy row from the source tile
    memcpy((void *)&a_map[i * kernel_x_shape_[1]], (void *)&a[i * a_shape_[1]],
           input_1_size * a_dtype_size_);
  }

  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // sync input activation to device memory
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_run_aie.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_size, 0);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ += static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ += static_cast<int64_t>(a_sync_stop - a_sync_start);
  for (int64_t cb = 0; cb < w_shape_[1]; cb += kernel_y_shape_[1]) {
    // compute current tile index along N dimension
    int64_t tile_col = cb / kernel_y_shape_[1];
    int64_t tile_idx = tile_col;

    if (wts_index >= 0) {
      tile_idx = wts_index;
    }

    auto kernel_ = xrt_ctx_->get_kernel();
    // launch the GEMM kernel
    auto run_aie_start = GET_ELAPSED_TIME_NS();
    if (is_bfp16_cast_gemm_) {
      // kernel call for GEMM that supports transaction binary flow
      auto run = kernel_(
          2, instr_bo, instr_bo_words,
          a_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
          weights_bo_[tile_idx].address() + DDR_AIE_ADDR_OFFSET,
          c_bo_run_aie.address() + DDR_AIE_ADDR_OFFSET,
          cast_ofm_bo_.address() + DDR_AIE_ADDR_OFFSET,
          (!cast_int4_ofm_bo_) // in bfp16 kernel if M<512, this bo is useless
              ? 0
              : cast_int4_ofm_bo_.address() + DDR_AIE_ADDR_OFFSET);
      if (wait) {
        run.wait2();
      }
    } else {
      ryzenai::dynamic_dispatch::execute_kernel(
          kernel_, 2, instr_bo, instr_bo_words, a_bo_run_aie,
          weights_bo_[tile_idx], c_bo_run_aie, c_bo_run_aie, 0, true, false);
    }
    auto run_aie_stop = GET_ELAPSED_TIME_NS();
    num_run_aie_++;
    run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);

    auto c_sync_start = GET_ELAPSED_TIME_NS();
    c_bo_run_aie.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                      a_shape_[0] * kernel_z_shape_[1] * sizeof(OutT), 0);
    auto c_sync_stop = GET_ELAPSED_TIME_NS();
    c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);
    auto c_copy_start = GET_ELAPSED_TIME_NS();

    // compute shape of current output tile
    int64_t output_shape[2];
    output_shape[1] = std::min(c_shape_[1] - cb, kernel_z_shape_[1]);

    // initialize the output tile
    auto c_map = c_bo_run_aie.map<AccT *>();
    // std::string golden_path = "bfp16_golden/data_" + std::to_string(2048) +
    // "_" +
    //                         std::to_string(4096) + "_" +
    //                         std::to_string(4096) + "/";
    // std::cout << "ofm after run" << std::endl;
    // auto ofm = read_file(golden_path + "ofm32_ref.txt");
    // uint16_t* ofm_ptr = (uint16_t*)(ofm.data());

    //   for ( int i =0 ;i < ofm.size() * 2 ; i++ ) {
    //     if (*(c_map +i) != *(ofm_ptr+i)) {
    //         std::cout << "ofm diff pos " << i << std::endl;
    //     }
    //   }
    //   std::cout << std::endl;

    for (int i = 0; i < a_shape_[0]; ++i) {
      memcpy((void *)&c_acc[(i)*c_shape_[1] + cb],
             (void *)&c_map[i * kernel_z_shape_[1]],
             output_shape[1] * sizeof(AccT));
    }

    auto c_copy_stop = GET_ELAPSED_TIME_NS();
    c_copy_time_ += static_cast<int64_t>(c_copy_stop - c_copy_start);
  }

  auto exec_end = GET_ELAPSED_TIME_NS();

  int64_t a_pad_time = 0;
  int64_t c_pad_time = 0;
  int64_t cpu_depad_time = 0;

  if (debug_) {
    // Write input / output matrix to file.
    std::string a_fname = "ryzenai_qlinear2_" +
                          std::to_string(mladfmatmulbias_id_) + "_" +
                          std::to_string(num_execute_) + "_tile_a.txt";
    std::string c_fname = "ryzenai_qlinear2_" +
                          std::to_string(mladfmatmulbias_id_) + "_" +
                          std::to_string(num_execute_) + "_tile_c.txt";

    Utils::write_buffer_to_file(a, a_shape_[0] * a_shape_[1], a_fname);
    Utils::write_buffer_to_file(c, c_shape_[0] * c_shape_[1], c_fname);
  }
  num_execute_++;

  RYZENAI_LOG_INFO(
      std::to_string(mladfmatmulbias_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_pad_time) + " " + std::to_string(c_pad_time) + " " +
      std::to_string(cpu_depad_time) + " " + std::to_string(a_copy_time_) +
      " " + std::to_string(a_sync_time_) + " " + std::to_string(c_copy_time_) +
      " " + std::to_string(c_sync_time_) + " " + std::to_string(cpu_acc_time_) +
      " " + std::to_string((double)run_aie_time_ / num_run_aie_) + " " +
      std::to_string(grp_size_) + "\n");
}

static std::tuple<size_t, size_t, size_t>
fit_MKN(const std::vector<Tensor> &input) {
  // input[0] --> input
  // input[1] --> wts
  size_t M = input.at(0).shape.size() == 3 ? input.at(0).shape.at(1)
                                           : input.at(0).shape.at(0);
  size_t K = input.at(1).shape.at(0);
  size_t N = input.at(1).shape.at(1);

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename AccT, typename OutT>
void mladfmatmulbias<InT, WtT, AccT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Matmul initialize_const_params(ptr) ...");
  if (is_bfp16_cast_gemm_) {
    int8_t *weights = (int8_t *)const_params.at(0).data;
    int8_t *zeros = (int8_t *)const_params.at(3).data;
    float *scales = (float *)const_params.at(2).data;
    float *bias = (float *)const_params.at(1).data;

    std::tuple<size_t, size_t> w_shape = {const_params.at(0).shape.at(0),
                                          const_params.at(0).shape.at(1)};
    w_shape_[0] = std::get<0>(w_shape);
    w_shape_[1] = std::get<1>(w_shape);
    int group_size = 32;
    std::string key = "group_size";
    if (attr.find(key) != attr.end()) {
      if (attr.at(key).type() == typeid(std::vector<int>)) {
        group_size = (std::any_cast<const std::vector<int> &>(attr.at(key)))[0];
      } else if (attr.at(key).type() == typeid(int)) {
        group_size = std::any_cast<int>(attr.find(key)->second);
      }
    }

    bfp16_grp_size_ = group_size;
    set_kernel_shapes_kn_mladf();
    kernel_x_shape_[0] = KERNEL_M_MAX;
    kernel_z_shape_[0] = KERNEL_M_MAX;
    w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
    w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);

    // wts(K,N), zp(K/GRP_SIZE,N), scale(K/GRP_SIZE,N), bias(1,N)
    // weights shape: (w_shape_[0], w_shape_[1]) -> (w_padded_shape_[0],
    // w_padded_shape_[1]) bias: w_shape_[1] -> w_padded_shape_[1] zeros:
    // (w_shape_[0]/group_size , w_shape_[1]) -> (w_padded_shape_[0]/group_size
    // , w_padded_shape_[1]) scales: (w_shape_[0]/group_size , w_shape_[1]) ->
    // (w_padded_shape_[0]/group_size , w_padded_shape_[1])
    std::vector<int8_t> pad_wts;
    std::vector<float> pad_bias;
    std::vector<float> pad_scales;
    std::vector<int8_t> pad_zeros;

    if (w_padded_shape_[1] > w_shape_[1] || w_padded_shape_[0] > w_shape_[0]) {
      pad_wts.resize(w_padded_shape_[0] * w_padded_shape_[1], 0);
      for (int i = 0; i < w_shape_[0]; i++) {
        for (int j = 0; j < w_shape_[1]; j++) {
          pad_wts[i * w_padded_shape_[1] + j] =
              *(weights + i * w_shape_[1] + j);
        }
      }
      weights = pad_wts.data();
      pad_bias.resize(w_padded_shape_[1], 0);
      for (int i = 0; i < w_shape_[1]; i++) {
        pad_bias[i] = *(bias + i);
      }
      bias = pad_bias.data();

      pad_scales.resize(w_padded_shape_[0] / group_size * w_padded_shape_[1],
                        0);
      for (int i = 0; i < w_shape_[0] / group_size; i++) {
        for (int j = 0; j < w_shape_[1]; j++) {
          pad_scales[i * w_padded_shape_[1] + j] =
              *(scales + i * w_shape_[1] + j);
        }
      }
      scales = pad_scales.data();

      pad_zeros.resize(w_padded_shape_[0] / group_size * w_padded_shape_[1], 0);
      for (int i = 0; i < w_shape_[0] / group_size; i++) {
        for (int j = 0; j < w_shape_[1]; j++) {
          pad_zeros[i * w_padded_shape_[1] + j] =
              *(zeros + i * w_shape_[1] + j);
        }
      }
      zeros = pad_zeros.data();
    }

    set_bfp16_kernel_shapes();
    xrt::bo b_bo;
    std::vector<uint8_t> bo_vec;

    WtT *bo_map = nullptr;
    auto b_format_start = GET_ELAPSED_TIME_NS();
    initialize_bfp16_wts(bo_vec, weights, zeros, bias, scales);

    io.write(0, bo_vec.data(), bo_vec.size());
    auto b_format_stop = GET_ELAPSED_TIME_NS();
    b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
  } else {
    DD_THROW_IF((const_params.size() != 4) ||
                    (const_params.at(0).shape.size() != 2),
                OpsFusion::dd_format("Unsupported const spec for Matmul\n") +
                    OpsFusion::dd_format("(Details : #const params == 1 "
                                         "({}), Const param dim == 2 ({})",
                                         const_params.size(),
                                         const_params.at(0).shape.size()));

    int8_t *weights = (int8_t *)const_params.at(0).data;
    int8_t *zeros = (int8_t *)const_params.at(3).data;
    float *scales = (float *)const_params.at(2).data;
    float *bias = (float *)const_params.at(1).data;
    std::tuple<size_t, size_t> w_shape = {const_params.at(0).shape.at(0),
                                          const_params.at(0).shape.at(1)};
    int group_size = 128;
    std::string key = "group_size";
    if (attr.at(key).type() == typeid(std::vector<int>)) {
      group_size = (std::any_cast<const std::vector<int> &>(attr.at(key)))[0];
    }
    // Note: for mladf int8 gemm we had to change group id to 0
    const int group_id = 0;

    w_shape_[0] = std::get<0>(w_shape);
    w_shape_[1] = std::get<1>(w_shape);
    set_kernel_shapes_kn_mladf();
    // Use largest M dimension as the default. This has to correspond
    // to one of the available kernel sizes.
    //    NOTE: smaller M's can be selected in run_aie if needed
    kernel_x_shape_[0] = KERNEL_M_MAX;
    kernel_z_shape_[0] = KERNEL_M_MAX;

    /* Create weight BOs */
    // Create a BO for weight block and initialize to zero
    //    NOTE: We must initialize to zero here because the weight matrix
    //          shape might not be an integer multiple of the block size.
    //          Initializing the BOs to zero takes care of the padding
    //          without allocating any extra scratch space.

    // For int4 quantization the buffer also contains bias, zeros, and scales
    // the weights are tiled in zigzag w4 aligned subvolumes of 32x128 tiles
    // the first subvolume consists of bias that is padded with zeros
    // Rest of the subvolumes consist weights+scales+zeros in each tile
    // QuantMatrix class has helper functions to write the data into the
    // correct index

    w_padded_shape_[0] = Utils::ceil_for_me(w_shape_[0], kernel_y_shape_[0]);
    w_padded_shape_[1] = Utils::ceil_for_me(w_shape_[1], kernel_y_shape_[1]);
    // The bfp16 kernel uses a block size of 4 for the default and 2 for the
    // updated overlay.
    int blk_size = op_version_ == "v1" ? 2 : 4;
    mladfQuantMatrix<64, 32, 32, 32> buff_B1((int)kernel_y_shape_[0],
                                             (int)kernel_y_shape_[1], blk_size);
    mladfQuantMatrix<64, 128, 32, 128> buff_B2(
        (int)kernel_y_shape_[0], (int)kernel_y_shape_[1], blk_size);

    // iterate over kernel shaped blocks of the weight matrix
    std::vector<uint8_t> bo_map;
    for (int64_t rb = 0; rb < w_padded_shape_[0]; rb += kernel_y_shape_[0]) {
      for (int64_t cb = 0; cb < w_padded_shape_[1]; cb += kernel_y_shape_[1]) {
        auto b_format_start = GET_ELAPSED_TIME_NS();

        int block_size =
            (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
        bo_map.resize(block_size);
        memset((void *)bo_map.data(), 0, block_size);

        buff_B1.data = (mladfCoreSubv<32, 32, 32> *)bo_map.data();
        buff_B2.data = (mladfCoreSubv<128, 32, 128> *)bo_map.data();

        // first pack the bias (bf16)
        for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; ++c) {
          if (rb == 0) {
            (group_size < 128)
                ? buff_B1.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c])
                : buff_B2.bias(c) = ryzenai::float_to_bfloat16(bias[cb + c]);
          }
        }
        // format quantized weights (int4/uint4)
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0]; ++r) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
               c += 2) {
            // NOTE: int8_t weights will be sign extended to int
            int x = weights[((rb + r) * w_shape_[1]) + (cb + c)];
            int y = weights[((rb + r) * w_shape_[1]) + (cb + c) + 1];
            if (b_dtype_ == "int4") {
              (group_size < 128)
                  ? buff_B1.quant(r, c) = ryzenai::pack_v2int4(x, y)
                  : buff_B2.quant(r, c) = ryzenai::pack_v2int4(x, y);
            } else {
              (group_size < 128)
                  ? buff_B1.quant(r, c) = ryzenai::pack_v2uint4(x, y)
                  : buff_B2.quant(r, c) = ryzenai::pack_v2uint4(x, y);
            }
          }
        }

        // Select the supported group_size
        if (group_size >= 128) {
          assert(group_size % 128 == 0 &&
                 "group_size should be div by 32 or 128");
          grp_size_ = 128;
        } else if (group_size >= 32) {
          assert(group_size % 32 == 0 &&
                 "group_size should be div by 32 or 128");
          grp_size_ = 32;
        }

        int repeat_count = group_size / grp_size_;
        // format the scales (bf16)
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
             r += group_size) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1]; c++) {
            for (int g = 0; g < repeat_count; g++) {
              (group_size < 128)
                  ? buff_B1.scale(r + g * grp_size_, c) =
                        ryzenai::float_to_bfloat16(scales[(
                            ((rb + r) * w_shape_[1] / group_size) + (cb + c))])
                  : buff_B2.scale(r + g * grp_size_, c) =
                        ryzenai::float_to_bfloat16(scales[(
                            ((rb + r) * w_shape_[1] / group_size) + (cb + c))]);
            }
          }
        }

        // format the zeros (int4)
        for (int r = 0; r < kernel_y_shape_[0] && rb + r < w_shape_[0];
             r += group_size) {
          for (int c = 0; c < kernel_y_shape_[1] && cb + c < w_shape_[1];
               c += 2) {
            size_t index = ((rb + r) * w_shape_[1] / (group_size)) + (cb + c);
            int x = zeros[index];
            int y = zeros[index + 1];
            int8_t pack_zeros;
            if (b_dtype_ == "int4") {
              pack_zeros = ryzenai::pack_v2int4(x, y);
            } else {
              pack_zeros = ryzenai::pack_v2uint4(x, y);
            }
            for (int g = 0; g < repeat_count; g++) {
              (group_size < 128)
                  ? buff_B1.zero(r + g * grp_size_, c) = pack_zeros
                  : buff_B2.zero(r + g * grp_size_, c) = pack_zeros;
            }
          }
        }
        io.write(0, bo_map.data(), block_size);
        auto b_format_stop = GET_ELAPSED_TIME_NS();
        b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
      }
    }
  }
  RYZENAI_LOG_TRACE("Matmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::vector<uint8_t>
mladfmatmulbias<InT, WtT, AccT, OutT>::generate_fused_txnbin(
    const mladf_matrix_shapes &tiling_info,
    const std::vector<int64_t> tiling_info_m, const int64_t &K,
    const int64_t &N, const int64_t &group_size) const {
  RYZENAI_LOG_TRACE("Generating tiling for tiling, num m tiles: " +
                    std::to_string(tiling_info_m.size()));
  std::vector<uint8_t> data;
  Transaction &txn = Transaction::getInstance();
  // get base transactions
  std::vector<std::vector<uint8_t>> base_transactions;
  base_transactions.reserve(tiling_info_m.size());
  for (const auto &m : tiling_info_m) {
    std::string txn_key = get_instr_key(txn_fname_prefix_, m, K, N, group_size);
    base_transactions.emplace_back(txn.get_txn_bvec(txn_key));
  }
  // put tiling in the right shape
  std::vector<std::vector<int64_t>> fused_tiling_info{
      tiling_info_m, std::vector<int64_t>(), std::vector<int64_t>()};
  // get input size and argmap info
  //  The bfp16 kernel uses a block size of 4 for the default and 2 for the
  //  updated overlay.
  int blk_size = op_version_ == "v1" ? 2 : 4;
  mladfQuantMatrix<64, 32, 32, 32> buff_B1((int)K, (int)N, blk_size);
  mladfQuantMatrix<64, 128, 32, 128> buff_B2((int)K, (int)N, blk_size);
  // iterate over kernel shaped blocks of the weight matrix
  int block_size = (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
  size_t const_params_bo_size = block_size;
  size_t input_bo_size = (1 * K * sizeof(InT));
  size_t output_bo_size = (1 * N * sizeof(OutT));
  bool needs_scratch = K > 4096;
  size_t scratch_bo_size = (needs_scratch) ? output_bo_size : 0;
  if (is_bfp16_cast_gemm_) {
    int sv_K = 64;
    int sv_N = 64;
    auto sv_K_num = K / sv_K;
    auto sv_N_num = N / sv_N;
    auto sv_size_in_byte =
        (sv_N * sv_K / 2) / 2 + (sv_N / 2 + 63) / 64 * 64 + sv_N * 2;

    size_t b_bo_size = sv_size_in_byte * 2 * (sv_K_num + 1) * (sv_N_num);
    size_t cast_ofm_bo_size = 36 * (K / 8) * (1 / 8) * 2;
    std::vector<OpArgMap> arg_map{
        {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
        {OpArgMap::OpArgType::CTRL_PKT_BIN, 3, 3, 0, cast_ofm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, b_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size}};
    data = matmul_nonuniform_tile_transaction_bin(base_transactions, arg_map,
                                                  fused_tiling_info);
  } else {
    if (needs_scratch) {
      std::vector<OpArgMap> arg_map{
          {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
          {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
          {OpArgMap::OpArgType::SCRATCH_PAD, 2, 4, 0, scratch_bo_size},
          {OpArgMap::OpArgType::OUTPUT, 3, 5, 0, output_bo_size}};
      data = matmul_nonuniform_tile_transaction_bin(base_transactions, arg_map,
                                                    fused_tiling_info);
    } else {
      std::vector<OpArgMap> arg_map{
          {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
          {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
          {OpArgMap::OpArgType::OUTPUT, 2, 5, 0, output_bo_size}};
      data = matmul_nonuniform_tile_transaction_bin(base_transactions, arg_map,
                                                    fused_tiling_info);
    }
  }
  return data;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::vector<uint8_t>
mladfmatmulbias<InT, WtT, AccT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<uint8_t> data;
  {
    auto [M, K, N] = fit_MKN(input);
    int group_size = 128;
    std::string key = "group_size";
    if (attr.at(key).type() == typeid(std::vector<int>)) {
      group_size = (std::any_cast<const std::vector<int> &>(attr.at(key)))[0];
    }
    auto [tiling_info, tiling_info_m, cost] = map_padded_shape(M, K, N);
    if (tiling_info_m.size() > 1) {
      data =
          generate_fused_txnbin(tiling_info, tiling_info_m, K, N, group_size);
    } else {
      Transaction &txn = Transaction::getInstance();
      std::string txn_key =
          get_instr_key(txn_fname_prefix_, M, K, N, group_size);
      data = txn.get_txn_bvec(txn_key);
    }
  }

  return data;
}

template <typename InT, typename WtT, typename AccT, typename OutT>
const std::vector<uint8_t>
mladfmatmulbias<InT, WtT, AccT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename InT, typename WtT, typename AccT, typename OutT>
std::vector<OpArgMap> mladfmatmulbias<InT, WtT, AccT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<OpArgMap> arg_map;
  // TODO: impl for mladfmatmulbias
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  w_shape_[0] = input.at(1).shape.at(0);
  w_shape_[1] = input.at(1).shape.at(1);
  set_kernel_shapes_kn_mladf();
  auto [tiling_info, tiling_info_m, cost] = map_padded_shape(
      input.at(0).shape.at(1), kernel_x_shape_[1], kernel_y_shape_[1]);
  kernel_x_shape_[0] = tiling_info.M;
  kernel_z_shape_[0] = tiling_info.M;

  // The bfp16 kernel uses a block size of 4 for the default and 2 for the
  // updated overlay.
  int blk_size = op_version_ == "v1" ? 2 : 4;
  mladfQuantMatrix<64, 32, 32, 32> buff_B1((int)kernel_y_shape_[0],
                                           (int)kernel_y_shape_[1], blk_size);
  mladfQuantMatrix<64, 128, 32, 128> buff_B2((int)kernel_y_shape_[0],
                                             (int)kernel_y_shape_[1], blk_size);
  // iterate over kernel shaped blocks of the weight matrix
  int group_size = 128;
  std::string key = "group_size";
  if (attr.at(key).type() == typeid(std::vector<int>)) {
    group_size = (std::any_cast<const std::vector<int> &>(attr.at(key)))[0];
  }
  int block_size = (group_size < 128) ? buff_B1.data_size : buff_B2.data_size;
  size_t const_params_bo_size = block_size;
  size_t input_bo_size = (tiling_info.K * kernel_x_shape_[0] * sizeof(InT));
  size_t output_bo_size = (tiling_info.N * kernel_z_shape_[0] * sizeof(OutT));

  bool needs_scratch = tiling_info.K > 4096;
  size_t scratch_bo_size = (needs_scratch) ? 2 * output_bo_size : 0;
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  if (is_bfp16_cast_gemm_) {
    auto cast_ofm_bo_size = static_cast<size_t>(36 * (kernel_y_shape_[0] / 8) *
                                                (kernel_x_shape_[0] / 8) * 2);
    int sv_K = 64;
    int sv_N = 64;
    auto sv_K_num = kernel_y_shape_[0] / sv_K;
    auto sv_N_num = kernel_y_shape_[1] / sv_N;
    auto sv_size_in_byte =
        (sv_N * sv_K / 2) / 2 + (sv_N / 2 + 63) / 64 * 64 + sv_N * 2;

    size_t b_bo_size = sv_size_in_byte * 2 * (sv_K_num + 1) * (sv_N_num);
    size_t cast_int4_ofm_bo_size = tiling_info.M < 512
                                       ? 0
                                       : 1152 * (kernel_y_shape_[0] / 64 + 1) *
                                             (kernel_y_shape_[1] / 32) * 2;

    std::vector<OpArgMap> arg_map{
        {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
        {OpArgMap::OpArgType::CTRL_PKT_BIN, 3, 3, 0, cast_ofm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, b_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 4, 4, 0, cast_int4_ofm_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size}};
    return arg_map;
  } else {
    if (needs_scratch) {
      arg_map = {
          {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
          {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
          {OpArgMap::OpArgType::SCRATCH_PAD, 3, 4, 0, scratch_bo_size},
          {OpArgMap::OpArgType::OUTPUT, 2, 5, 0, output_bo_size}};

      return arg_map;
    } else {
      arg_map = {
          {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
          {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
          {OpArgMap::OpArgType::OUTPUT, 2, 5, 0, output_bo_size}};

      return arg_map;
    }
  }
}
template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag mladfmatmulbias<InT, WtT, AccT, OutT>::logger_flag_;
template <typename InT, typename WtT, typename AccT, typename OutT>
uint64_t mladfmatmulbias<InT, WtT, AccT, OutT>::mladfmatmulbias_count = 0;

template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag mladfmatmulbias<InT, WtT, AccT, OutT>::instr_reg_flag_;
template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag mladfmatmulbias<InT, WtT, AccT, OutT>::instr_reg_v1_flag_;
template <typename InT, typename WtT, typename AccT, typename OutT>
std::once_flag mladfmatmulbias<InT, WtT, AccT, OutT>::supported_shapes_flag_;

template <typename InT, typename WtT, typename AccT, typename OutT>
std::mutex mladfmatmulbias<InT, WtT, AccT, OutT>::instr_reg_mutex_;

template class mladfmatmulbias<int16_t, uint8_t, int16_t, int16_t>;
template class mladfmatmulbias<int16_t, int8_t, int16_t, int16_t>;
template class mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>;
template class mladfmatmulbias<uint16_t, uint8_t, uint16_t, uint16_t>;
template class mladfmatmulbias<uint16_t, uint16_t, int16_t, int16_t>;
} // namespace ryzenai
