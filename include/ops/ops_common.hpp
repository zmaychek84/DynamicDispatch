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
#include <tuple>
#include <vector>

namespace ryzenai {
struct matrix_shapes {
  // capture M, K, N of the shape supported.
  int64_t M;
  int64_t K;
  int64_t N;

  matrix_shapes(int64_t M, int64_t K, int64_t N) : M(M), K(K), N(N) {}
  matrix_shapes() : M(1), K(1), N(1) {}
};

struct conv_shapes {
  /* This shape only supports square filter dimention i.e. FxF */
  // capture zp, F, K, N of the shape supported.
  int64_t Z; /* Zero point */
  int64_t F; /* Filter size : Typically F x F*/
  int64_t K; /* Number of input channels */
  int64_t N; /* Number of output channels */

  conv_shapes(int64_t Z, int64_t F, int64_t K, int64_t N)
      : Z(Z), F(F), K(K), N(N) {}
};

struct sd_conv2d_shapes {
  /* this shape definition is same as conv2d shape in pytorch */
  int64_t OC; // output channel
  int64_t IC; // input channel
  int64_t IH; // input height
  int64_t IW; // input width
  int64_t OH; // input height
  int64_t OW; // input width
  int64_t kh; // kernel height
  int64_t kw; // kernel width

  sd_conv2d_shapes(int64_t OC, int64_t IC, int64_t IH, int64_t IW, int64_t OH,
                   int64_t OW, int64_t kh, int64_t kw)
      : OC(OC), IC(IC), IH(IH), IW(IW), OH(OH), OW(OW), kh(kh), kw(kw) {}
};

struct mladf_matrix_shapes {
  // capture M, K, N, Gs of the shape supported.
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t Gs;

  mladf_matrix_shapes(int64_t M, int64_t K, int64_t N, int64_t Gs = 0)
      : M(M), K(K), N(N), Gs(Gs) {}
  mladf_matrix_shapes() : M(1), K(1), N(1), Gs(1) {}
};

const std::string mdsqr_A8W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psf_model_a8w8_qdq.xclbin";
const std::string mdsqrv1_1_A8W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psf_v1.1_model_a8w8_qdq.xclbin";
const std::string mxpzi_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psj_model_a16w8_qdq.xclbin";
const std::string mxgan_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psh_model_a16w8_qdq.xclbin";
const std::string mxganv1_2_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psh_v1.2_model_a16w8_qdq.xclbin";
const std::string m3uec_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psi_model_a16w8_qdq.xclbin";
const std::string mtea0a_A8W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psq_model_a8w8_qdq.xclbin";
const std::string m7h4xjg_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psq2_model_a16w8_qdq.xclbin";
const std::string mzdk5_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_psr_model_a16w8_qdq.xclbin";
const std::string PSS_A16A16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin";
const std::string PST_A16A16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_pss_a16a16_qdq.xclbin";
const std::string MLADF_4x4_GEMM_SILU_MUL_A16FW4_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x4_gemm_silu_mul_a16fw4.xclbin";
const std::string mzdk54x4_A16W8_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x4_psr_model_a16w8_qdq.xclbin";
const std::string MLADF_GEMM_4x4_A16FW4ACC16F_XCLBIN_PATH =
    "/xclbin/stx/mladf_gemm_4x4_a16fw4acc16f.xclbin";
const std::string MLADF_2x4x4_MASKEDSOFTMAX_A16F_XCLBIN_PATH =
    "/xclbin/stx/mladf_2x4x4_maskedsoftmax_a16f.xclbin";
const std::string BMM_A16W16_65536_128_2048_XCLBIN_PATH =
    "/xclbin/stx/2x4x4_bmm_model_a16w16_65536_128_2048.xclbin";
const std::string BMM_A16W16_65536_2048_128_XCLBIN_PATH =
    "/xclbin/stx/2x4x4_bmm_model_a16w16_65536_2048_128.xclbin";
const std::string MLADF_4x2_GEMM_A16W8_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_gemm_a16w8_qdq.xclbin";
const std::string MLADF_4x2_GEMM_A16W16_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_gemm_a16w16_qdq.xclbin";
const std::string BMM_A16W16_XCLBIN_PATH =
    "/xclbin/stx/2x4x4_bmm_model_a16w16.xclbin";
const std::string XCOM_4x4_XCLBIN_PATH = "/xclbin/stx/4x4_dpu.xclbin";
const std::string XCOM_4x4_Q_XCLBIN_PATH =
    "/xclbin/stx/4x4_dpu_qconv_qelew_add.xclbin";
const std::string MLADF_SOFTMAX_A16_XCLBIN_PATH =
    "/xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin";
const std::string
    LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH =
        "/xclbin/stx/llama2_mladf_2x4x4_gemmbfp16_silu_mul_mha_rms_rope.xclbin";
const std::string
    LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH =
        "/xclbin/stx/"
        "llama2_mladf_2x4x4_v1_gemmbfp16_silu_mul_mha_rms_rope.xclbin";
const std::string MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_4x2_add_a16.xclbin";
const std::string MLADF_ELWMUL_A16_QDQ_XCLBIN_PATH =
    "/xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin";
const std::string START_TAIL_4x2_MS_SHELL_QDQ_XCLBIN_PATH =
    "/xclbin/stx/4x2_ps_start_tail_ops_qdq.xclbin";

namespace utils {
template <class Tuple,
          class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> tuple_to_vector(Tuple &&tuple) {
  return std::apply(
      [](auto &&...elems) {
        return std::vector<T>{std::forward<decltype(elems)>(elems)...};
      },
      std::forward<Tuple>(tuple));
}

template <class Types>
Types running_product_with_skips(const std::vector<Types> &nums,
                                 const std::vector<size_t> &skip_indexes = {}) {
  Types product_of_all =
      std::accumulate(nums.begin(), nums.end(), static_cast<Types>(1),
                      std::multiplies<Types>());
  Types product_of_skips{1};
  if (skip_indexes.size() != 0) {
    product_of_skips =
        std::accumulate(skip_indexes.begin(), skip_indexes.end(),
                        static_cast<Types>(1), [&](Types acc, size_t index) {
                          return index < nums.size() ? acc * nums[index] : acc;
                        });
  }

  return product_of_all / product_of_skips;
}

template <class Types>
Types max_element_count_with_skips(
    const std::vector<std::vector<Types>> &supported_shapes,
    const std::vector<size_t> skip_indexes = {}) {
  auto max_product_iter =
      max_element(supported_shapes.begin(), supported_shapes.end(),
                  [&](const auto &t1, const auto &t2) {
                    return running_product_with_skips(t1, skip_indexes) <
                           running_product_with_skips(t2, skip_indexes);
                  });
  return running_product_with_skips(*max_product_iter, skip_indexes);
}

inline int to_next_multiple(int number, int multiple) {
  return number + (multiple - number % multiple);
}

} // namespace utils
} // namespace ryzenai
