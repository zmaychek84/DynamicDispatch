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

#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
// #include <ops/maskedsoftmax/maskedsoftmax.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

// #include "maskedsoftmax_helpers.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <ops/matmul_cpu/matmul_cpu.hpp>

using namespace matmul_matrix;

template <typename T, typename T2>
int check_result_deqlinear(T cpu_Y, T aie_Y, T2 a,
                           bool enable_logging = false) {
  int fail = 0;
  int err_count = 0;
  float max_diff = 0;
  float L2_norm = 0;
  for (int r = 0; r < aie_Y.num_rows; ++r) {
    for (int c = 0; c < aie_Y.num_cols; ++c) {
      float diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > 0.01f) {
        if (enable_logging) {
          std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                    << "Expected: " << float(cpu_Y.at(r, c)) << ", "
                    << "Received: " << float(aie_Y.at(r, c)) << ", "
                    << "in: " << int(a[c]) << ", "
                    << "Diff: " << int(diff) << "\n";
        }
        fail = 1;
        err_count++;
      }
    }
  }
  // L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  // std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

template <typename T> T generateRandomInt() {
  static const int max = std::numeric_limits<T>::max();
  static const int min = std::numeric_limits<T>::min();
  return min + rand() % (max - min + 1);
}

template <typename T>
void dequantize_ref(T *quantized_data, float *dequantized_data,
                    const size_t size, const float scale,
                    const int32_t zero_point) {
  for (size_t i = 0; i < size; ++i) {
    dequantized_data[i] =
        (static_cast<int32_t>(quantized_data[i]) - zero_point) * scale;
  }
}

template <typename InT = uint16_t, typename OuT = float>
static int test_dequantizelinear_cpu(const std::string &meta_json, size_t B,
                                     size_t M, size_t K, bool debug = false,
                                     const std::string &a_dtype = "uint16",
                                     const std::string &c_dtype = "float") {
  // TODO
  // start of duplicated code from unit test
  std::vector<size_t> a_shape = {B, M, K};

  size_t tensor_sz = B * M * K;
  std::vector<InT> a(tensor_sz);

  // feed random float data
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = generateRandomInt<InT>();
  }
  // Range taken from
  // https://gitenterprise.xilinx.com/AIELibs/mllib/blob/dev/internal/models/python/restructured/operators/Transformers/SoftMax.py#L348
  // dd::initialize_random_float(a, 5);

  std::vector<OuT> aie_out(tensor_sz, garbage_value);
  std::vector<OuT> ref_out(tensor_sz, garbage_value);
  float sc_ = 0.0487f;
  int zp_ = 24;
  dequantize_ref(a.data(), ref_out.data(), tensor_sz, sc_, zp_);

  std::string xclbin_fname;
  std::string bksl = "\\";
  if (c_dtype == "uint16") {
    xclbin_fname =
        Utils::get_env_var("DD_ROOT") + bksl + mxpzi_A16W8_QDQ_XCLBIN_REL_PATH;
  } else {
    xclbin_fname =
        Utils::get_env_var("DD_ROOT") + bksl + mdsqr_A8W8_QDQ_XCLBIN_REL_PATH;
  }
  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{aie_out.data(), a_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);
  RowMajorMatrix<OuT> ref(1, tensor_sz, ref_out.data());
  RowMajorMatrix<OuT> cpu_out(1, tensor_sz, aie_out.data());
  int err_count = check_result_deqlinear(cpu_out, ref, a.data());

  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_dequantizelinear_cpu.exe <meta.json>"
              << std::endl;
    return EXIT_FAILURE;
  }
  size_t B = 12;
  size_t M = 64;
  size_t N = 512;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_dequantizelinear_cpu(meta_json, B, M, N, false);
    if (err_count > 0) {
      std::cout << "Dequantizelinear CPU test failed with err_count = "
                << err_count << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
