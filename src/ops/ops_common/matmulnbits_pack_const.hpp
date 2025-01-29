/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <any>
#include <assert.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "ops/op_const_io.hpp"

namespace nb = nanobind;

nb::ndarray<nb::numpy, const uint8_t>
matmulnbits_pack_const_float32(nb::ndarray<> weights, nb::ndarray<> bias,
                               nb::ndarray<> scale, nb::ndarray<> zeros, int K,
                               int N, int block_size, bool bias_en,
                               bool asymmetric_quant) {

  const size_t kblks = (K + block_size - 1) / block_size;
  std::vector<float> npu_scales(N * kblks);
  std::vector<int8_t> npu_weights(N * kblks * block_size, 0);
  // fill this with zeros for Symmetric quantization
  std::vector<int8_t> npu_zeros(N * kblks, 0);
  // fill this with zeros for MatMul without bias
  std::vector<float> npu_bias(N, 0);

  // Original weights are in NxK/2 packed as uint8
  // Convert to KXN uint8
  const uint8_t *wts = (const uint8_t *)weights.data();
  for (int64_t i = 0; i < K; i += 2) {
    for (int64_t j = 0; j < N; j++) {
      auto srcv = wts[j * K / 2 + i / 2];
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      npu_weights[i * N + j] = static_cast<int8_t>(src0);
      npu_weights[(i + 1) * N + j] = static_cast<int8_t>(src1);
    }
  }

  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BlockSize)xN shape
  const float *scl = (const float *)scale.data();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < kblks; j++) {
      npu_scales[j * N + i] = scl[i * kblks + j];
    }
  }

  // fill this with zeros for Symmetric quantization
  if (asymmetric_quant) {

    // Input zero_points is stored as uint8_t or same as type(A)
    // It has the same packing method as input B. - [N *
    // CeilDiv(n_blocks_per_col * bits, 8)] If zero_points has same type as A,
    // it's not packed and has the same shape as Scales
    // constexpr int32_t BITS = 4;
    // In our case, if kblks is odd, there will be extra 4-bits padding in a row
    const size_t kblks_zp = ((kblks + 1) / 2) * 2;
    const uint8_t *zero_pt = (const uint8_t *)zeros.data();
    for (int i = 0; i < N; i++) {
      // in src, each byte will have (up to) 2 zero points
      for (int j = 0; j < kblks; j = j + 2) {
        auto zpv = zero_pt[(i * (kblks_zp / 2)) + (j / 2)];
        npu_zeros[j * N + i] = (zpv & 0xf) - 8;

        if ((j + 1) < kblks) {
          npu_zeros[(j + 1) * N + i] = ((zpv & 0xf0) >> 4) - 8;
        }
      }
    }
  }

  if (bias_en) {
    const float *m_bias_ptr = (const float *)bias.data();
    for (int i = 0; i < N; i++) {
      npu_bias[i] = m_bias_ptr[i];
    }
  }

  // Weights shape
  std::vector<size_t> weight_shape = {static_cast<size_t>(kblks * block_size),
                                      static_cast<size_t>(N)};
  // Constant tensors
  Tensor weight_tensor = {npu_weights.data(), weight_shape, "int4"};
  // TODO: why is block_size here and not 1 x N?
  Tensor bias_tensor = {npu_bias.data(), {(size_t)block_size, 0}, "float"};
  // TODO: why is block_size here and not kblks x N?
  Tensor scales_tensor = {npu_scales.data(), {(size_t)block_size, 0}, "float"};
  // TODO: why is KxN here and not kblks x N?
  Tensor zeros_tensor = {npu_zeros.data(), weight_shape, "int4"};
  std::vector<Tensor> constant_tensors = {weight_tensor, bias_tensor,
                                          scales_tensor, zeros_tensor};

  std::string mladf_version_("v1");
  constexpr int kMaxSeqLength = 3072;

  std::map<std::string, std::any> attrs;
  attrs["op_version"] = mladf_version_;

  auto ptr = std::make_unique<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", false, attrs);

  attrs["default_shape"] = 1;
  attrs["max_m"] = kMaxSeqLength;
  attrs["group_size"] = static_cast<int>(block_size);
  std::vector<std::vector<uint8_t>> const_vecs =
      ptr->export_const_params(constant_tensors, attrs);

  if (const_vecs.size() != 1) {
    throw std::runtime_error("Expect to get 1 constant tensor");
  }

  if (const_vecs.at(0).size() == 0) {
    throw std::runtime_error("Expect to get non-empty constant tensor");
  }

  const size_t total_bytes = const_vecs.at(0).size();

  uint8_t *out_array = new uint8_t[total_bytes];
  std::memcpy(out_array, const_vecs.at(0).data(), total_bytes);
  size_t shape[1] = {total_bytes};

  return nb::ndarray<nb::numpy, const uint8_t>(
      /* data = */ out_array,
      /* ndim = */ 1,
      /* shape pointer = */ shape,
      /* owner = */ nb::handle());
}
