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

#include <iostream>
#include <torch/torch.h>

#include "ops/ops_common/matrix_formatting.h"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>

// #define UNIT_TEST_PERF

#include "enable_perf.hpp"

#include "test_common.hpp"

inline float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

template <typename T>
static int check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
                        float error_tolerance = 0.01,
                        float pixel_L2_norm_tolerance = 0.01) {

  int fail = 0;
  float max_diff = 0;
  float L2_norm = 0;
  int err_count = 0;
  for (int i = 0; i < cpu_Y.size(); ++i) {
    float diff = std::abs(bfloat16_to_float(cpu_Y.at(i)) -
                          bfloat16_to_float(aie_Y.at(i)));

    L2_norm += (diff * diff);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > error_tolerance) {
      // if (err_count < 100) {
      //   std::cout << "ERROR: Y[" << i << "]: "
      //             << "Expected: " << bfloat16_to_float(cpu_Y.at(i)) << ","
      //             << "Received: " << bfloat16_to_float(aie_Y.at(i))
      //             << " abs diff " << diff << " cumulative l2norm " << L2_norm
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

    std::cout << "deem err_count (" << err_count
              << ") as zero due to low pixel_L2_norm" << std::endl;
    err_count = 0;
  }
  std::cout << "err_count is " << err_count << std::endl;
  return err_count;
}

template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size, int data_max,
                              std::string dtype = "int8") {
  auto data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    if (dtype == "bfloat16") {
      vec[i] = ryzenai::rand_bfloat16(float(data_max));
    } else if (dtype == "uint4") {
      vec[i] = ryzenai::rand_uint4(data_max);
    } else if (dtype == "int4") {
      vec[i] = ryzenai::rand_int4(data_max);
    } else if (std::is_same<T, float>::value) {
      vec[i] = (2.0 * (rand() / (float)RAND_MAX) - 1.0) * data_max;
    } else {
      vec[i] = (T)(rand() % (data_max - data_min + 1)) + data_min;
    }
  }
}

static std::vector<uint32_t> read_file(const std::string &filePath) {
  std::ifstream fileStream(filePath);
  // std::cerr << "ut: try to open file " << filePath << std::endl;
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

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_matmul_mladf(int M, int K, int N, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "int4",
                      const std::string &c_dtype = "bfloat16",
                      int group_size = 128, bool compare_values = true,
                      std::string op_version = "v1") {

  if (b_dtype == "int4" || b_dtype == "uint4") {
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};
    std::vector<InT> a(M * K);
    std::vector<float> bias(N);
    std::vector<float> scales(K * N / group_size);
    std::vector<WgT> b(K * N);
    std::vector<WgT> zeros(K * N / group_size);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    // std::vector<float> c_golden(M * N, garbage_value);
    srand(42);
    std::map<std::string, std::any> attr;

    attr["op_version"] = op_version;
    attr["group_size"] = group_size;
    attr["is_gemm_cast"] = true;
    // Select the input data range for activations, weights, scales, and bias
    initialize_random<InT>(a, M * K, 100, "bfloat16");
    initialize_random<WgT>(b, K * N, 7, b_dtype);
    initialize_random<WgT>(zeros, K * N / group_size, 7, b_dtype);
    initialize_random<float>(bias, N, 1);
    initialize_random<float>(scales, K * N / group_size, 1);

    ryzenai::mladfmatmulbias mladfmatmulbias_ =
        ryzenai::mladfmatmulbias<InT, WgT, OuT, OuT>(a_dtype, b_dtype, c_dtype,
                                                     true, attr);

    mladfmatmulbias_.debug(debug);

    size_t Ms = static_cast<size_t>(M);
    size_t Ks = static_cast<size_t>(K);
    size_t Ns = static_cast<size_t>(N);

    std::vector<Tensor> const_Tensor;
    std::vector<size_t> v_shape_vec = {Ms, Ks};
    std::vector<size_t> b_shape_vec = {Ks, Ns};
    std::vector<size_t> size_shape = {0, 0}; // useless here
    const_Tensor = {{b.data(), b_shape_vec, b_dtype},
                    {bias.data(), size_shape, a_dtype},
                    {scales.data(), size_shape, a_dtype},
                    {zeros.data(), b_shape_vec, b_dtype}};

    std::vector<Tensor> input_Tensor;
    std::vector<size_t> a_shape_vec = {Ms, Ks};

    input_Tensor = {{a.data(), a_shape_vec, a_dtype}};

    std::vector<Tensor> output_Tensor;
    std::vector<size_t> c_shape_vec = {Ms, Ns};
    output_Tensor = {{c.data(), c_shape_vec, c_dtype}};
    mladfmatmulbias_.set_shape(a_shape_vec, b_shape_vec, group_size);
    mladfmatmulbias_.initialize_const_params(const_Tensor, attr);

#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N
                    << ", Gs = " << group_size);
    PROFILE_THIS(mladfmatmulbias_.execute(input_Tensor, output_Tensor));
#else
    mladfmatmulbias_.execute(input_Tensor, output_Tensor);
#endif
    if (!compare_values) {
      return 0;
    }

    // Compute using pytorch
    torch::Tensor a_f = torch::empty({M, K});
    torch::Tensor b_f = torch::empty({K, N});
    torch::Tensor bias_f = torch::empty({1, N});

    // convert bfloat16 activation to float
    std::vector<float> a_f_vec(M * K);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        a_f_vec[m * K + k] = ryzenai::bfloat16_rnd_even(
            ryzenai::bfloat16_to_float(a[m * K + k]));
      }
    }
    a_f = torch::from_blob(a_f_vec.data(), {M, K}, torch::kFloat32);

    // set bias
    bias_f = torch::from_blob(bias.data(), {1, N}, torch::kFloat32);

    // dequantize weight
    std::vector<float> b_f_vec(K * N);
    for (int n = 0; n < N; ++n) {
      // bias_f[0][n] = bias[n];
      for (int k = 0; k < K; ++k) {
        int g_idx = (k / group_size);
        b_f_vec[k * N + n] =
            (b[k * N + n] - zeros[g_idx * N + n]) * scales[g_idx * N + n];
      }
    }
    b_f = torch::from_blob(b_f_vec.data(), {K, N}, torch::kFloat32);

    // Calculate MatMulBias
    auto ret = torch::matmul(a_f, b_f);
    ret = ret + bias_f;

    float *c_golden = ret.data_ptr<float>();

    float const EPSILON_MAX =
        7.0; // this is the tolerated max error, normalized by sqrt(K)
    float const EPSILON_MEAN =
        0.8; // this is the tolerated mean error, normalized by sqrt(K)
    float const EPSILON_SCALE =
        b_dtype == "int4"
            ? 2.5
            : 1; // for int4 weights, the quantization deviations are higher
    int err_count = 0;
    float err_max = 0;
    float err_min = 0;
    float err_total = 0;
    float err_mean = 0;

    for (int i = 0; i < c.size(); i++) {
      float err = std::abs(ryzenai::bfloat16_rnd_even(c_golden[i]) -
                           ryzenai::bfloat16_to_float(c[i]));
      if (std::abs(err_max) < std::abs(err)) {
        err_max = err;
      }
      if (i == 0) {
        err_min = err;
      } else if (std::abs(err_min) > std::abs(err)) {
        err_min = err;
      }
      err_total += err;
      if (err > EPSILON_MAX * sqrt(K) * EPSILON_SCALE) {
        err_count++;
        if (err_count < 16) {
          std::cout << "First deviating values:"
                    << "\n";
          std::cout << std::dec << "c[" << i << "]: "
                    << "Err: " << err << ", "
                    << "Expected: " << ryzenai::bfloat16_rnd_even(c_golden[i])
                    << ", "
                    << "Received: " << ryzenai::bfloat16_to_float(c[i]) << "\n";
        }
      }
    }

    err_mean = err_total / c.size();
    printf("err_max: %.2f, target: %.2f\n", err_max,
           EPSILON_MAX * sqrt(K) * EPSILON_SCALE);
    printf("err_mean: %.2f, target: %.2f\n", err_mean,
           EPSILON_MEAN * sqrt(K) * EPSILON_SCALE);

    if (err_count > 0) {
      std::cout << std::dec << std::fixed << std::setprecision(3)
                << 100.0 * err_count / c.size()
                << "\% of the values deviate more than allowed." << std::endl;
    }
    bool max_error_violation =
        std::isnan(err_max) || err_max > EPSILON_MAX * sqrt(K) * EPSILON_SCALE;
    bool mean_error_violation =
        std::isnan(err_mean) ||
        err_mean > EPSILON_MEAN * sqrt(K) * EPSILON_SCALE;
    return max_error_violation || mean_error_violation;
  }
}

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_bfp16_cast_gemm_mladf(int M, int K, int N, bool debug = false,
                               const std::string &a_dtype = "bfloat16",
                               const std::string &b_dtype = "uint4",
                               const std::string &c_dtype = "bfloat16",
                               int group_size = 128, bool compare_values = true,
                               std::string op_version = "bfp16gemm") {
  int err_count = 1;
  if (b_dtype == "uint4") {
    std::cerr << "M " << M << " K " << K << " N " << N << std::endl;
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};
    std::vector<InT> a(M * K);
    std::vector<WgT> b(K * N / 2);
    std::vector<WgT> zeros(K * N / group_size / 2);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    // std::vector<float> c_golden(M * N, garbage_value);
    srand(42);
    std::map<std::string, std::any> attr;

    attr["op_version"] = op_version;
    attr["group_size"] = group_size;
    attr["is_gemm_cast"] = true;

    // Select the input data range for activations, weights, scales, and bias
    std::string golden_path = "bfp16_golden/data_" + std::to_string(M) + "_" +
                              std::to_string(K) + "_" + std::to_string(N) + "/";

    auto ifm = read_file(golden_path + "ifm32.txt");

    auto wts = read_file(golden_path + "wts.txt");
    auto bias = read_file(golden_path + "bias.txt");
    auto zp = read_file(golden_path + "zp.txt");
    auto scales = read_file(golden_path + "scale.txt");

    // std::vector<uint8_t> bo_map;
    // std::cout << "wts.size() " << wts.size() * 4 << std::endl;
    // std::cout << "zp.size() " << zp.size() * 4 << std::endl;
    // std::cout << "scales.size() " << scales.size() << std::endl;
    // std::cout << "bias.size() " << bias.size() << std::endl;
    // std::cout << "ifm bytes " << ifm.size() * 4 << std::endl;
    // std::cout << op_version << std::endl;

    ryzenai::mladfmatmulbias mladfmatmulbias_ =
        ryzenai::mladfmatmulbias<InT, WgT, OuT, OuT>(a_dtype, b_dtype, c_dtype,
                                                     true, attr);

    mladfmatmulbias_.debug(debug);

    size_t Ms = static_cast<size_t>(M);
    size_t Ks = static_cast<size_t>(K);
    size_t Ns = static_cast<size_t>(N);

    std::vector<Tensor> const_Tensor;
    std::vector<size_t> v_shape_vec = {Ms, Ks};
    std::vector<size_t> b_shape_vec = {Ks, Ns};
    std::vector<size_t> size_shape = {0, 0}; // useless here
    const_Tensor = {{wts.data(), b_shape_vec, b_dtype},
                    {bias.data(), size_shape, "float"},
                    {scales.data(), size_shape, "float"},
                    {zp.data(), size_shape, b_dtype}};

    std::vector<Tensor> input_Tensor;
    std::vector<size_t> a_shape_vec = {Ms, Ks};

    input_Tensor = {{ifm.data(), a_shape_vec, a_dtype}};

    std::vector<Tensor> output_Tensor;
    std::vector<size_t> c_shape_vec = {Ms, Ns};
    output_Tensor = {{c.data(), c_shape_vec, c_dtype}};
    mladfmatmulbias_.set_shape(a_shape_vec, b_shape_vec, group_size);
    mladfmatmulbias_.initialize_const_params(const_Tensor, attr);
    std::cout << "execute before" << std::endl;
#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N
                    << ", Gs = " << group_size);
    PROFILE_THIS(mladfmatmulbias_.execute(input_Tensor, output_Tensor));
#else
    mladfmatmulbias_.execute(input_Tensor, output_Tensor);
#endif

    auto output_golden = read_file("bfp16_golden/data_" + std::to_string(M) +
                                   "_" + std::to_string(K) + "_" +
                                   std::to_string(N) + "/ofm32_ref.txt");

    // std::cout << " output_golden byte size " << output_golden.size() * 4
    //           << std::endl;
    // std::cout << " c byte size " << c.size() * 2 << std::endl;
    std::vector<OuT> bf16_output_golden(c.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = check_result(bf16_output_golden, c);
  }

  return err_count;
}

std::vector<float> read_npy_float(const std::string &filename) {
  std::cout << "open file " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file.");
  }

  char header[6];
  file.read(header, 6);
  if (std::memcmp(header, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Not a valid .npy file.");
  }

  file.seekg(128, std::ios::beg);

  std::vector<float> data;
  float value;
  while (file.read(reinterpret_cast<char *>(&value), sizeof(float))) {
    data.push_back(value);
  }

  file.close();
  return data;
}

std::vector<uint8_t> read_npy_int8(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  char magic[6];
  file.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    throw std::runtime_error("Invalid .npy file.");
  }

  uint8_t major, minor;
  file.read(reinterpret_cast<char *>(&major), 1);
  file.read(reinterpret_cast<char *>(&minor), 1);

  uint16_t header_len;
  file.read(reinterpret_cast<char *>(&header_len), 2);
  if (major > 1) {
    header_len = (header_len & 0xFF) | (header_len >> 8);
  }

  std::string header(header_len, ' ');
  file.read(header.data(), header_len);

  std::string dtype = "'descr': '<i1'";
  if (header.find(dtype) == std::string::npos) {
    throw std::runtime_error("Unsupported data type or endianness.");
  }

  std::string shape_key = "'shape': (";
  size_t shape_pos = header.find(shape_key);
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("Shape information not found in header.");
  }

  size_t shape_start = shape_pos + shape_key.length();
  size_t shape_end = header.find(')', shape_start);
  if (shape_end == std::string::npos) {
    throw std::runtime_error("Malformed shape information.");
  }

  std::string shape_str = header.substr(shape_start, shape_end - shape_start);
  size_t total_elements = 1;
  size_t pos = 0;
  while ((pos = shape_str.find(',')) != std::string::npos) {
    total_elements *= std::stoul(shape_str.substr(0, pos));
    shape_str = shape_str.substr(pos + 1);
  }
  if (!shape_str.empty()) {
    total_elements *= std::stoul(shape_str);
  }

  std::vector<uint8_t> data(total_elements);
  file.read(reinterpret_cast<char *>(data.data()), total_elements);

  if (!file) {
    throw std::runtime_error("File reading error.");
  }

  file.close();
  return data;
}

TEST(Bfp16Gemm, TestWtsShuffle) {
  uint64_t M = 2048;
  uint64_t K = 4096;
  uint64_t N = 4096;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape_vec = {Ms, Ks};
  std::vector<size_t> b_shape_vec = {Ks, Ns};

  std::map<std::string, std::any> attr;
  std::vector<int> input_shape = {int(M), int(K), int(N)};
  attr["input_shape"] = input_shape;

  attr["op_version"] = std::string("bfp16gemm");
  attr["group_size"] = 128;
  attr["is_gemm_cast"] = true;
  auto mladfmatmulbias_ =
      ryzenai::mladfmatmulbias<uint16_t, uint8_t, uint16_t, uint16_t>(
          "bfloat16", "uint4", "bfloat16", true, attr);
  mladfmatmulbias_.set_shape(a_shape_vec, b_shape_vec, 128);
  std::vector<Tensor> const_tensors;
  std::string wts_shuffle_test_dir = "bfp16_golden/data_" + std::to_string(M) +
                                     "_" + std::to_string(K) + "_" +
                                     std::to_string(N) + "/";
  // NxK int8
  auto wts = read_file(wts_shuffle_test_dir + "wts.txt");
  auto bias = read_file(wts_shuffle_test_dir + "bias.txt");
  auto zp = read_file(wts_shuffle_test_dir + "zp.txt");
  auto scales = read_file(wts_shuffle_test_dir + "scale.txt");
  auto wts_aie_golden = read_file(wts_shuffle_test_dir + "wts32.txt");

  std::vector<uint8_t> bo_map;
  std::cout << "wts.size() " << wts.size() * 4 << std::endl;
  std::cout << "zp.size() " << zp.size() * 4 << std::endl;
  std::cout << "scales.size() " << scales.size() << std::endl;
  std::cout << "bias.size() " << bias.size() << std::endl;
  mladfmatmulbias_.set_bfp16_kernel_shapes();
  mladfmatmulbias_.initialize_bfp16_wts(
      bo_map, reinterpret_cast<int8_t *>(wts.data()),
      reinterpret_cast<int8_t *>(zp.data()),
      reinterpret_cast<float *>(bias.data()),
      reinterpret_cast<float *>(scales.data()));
  // std::vector<uint32_t> wts_aie = read_file(wts_shuffle_test_dir +
  // "wts32.txt");
  std::cout << "bo_map.size() " << bo_map.size() << std::endl;
  std::cout << "wts_aie_golden.size() " << wts_aie_golden.size() << std::endl;
  uint8_t *wts_aie_uint8_ptr =
      reinterpret_cast<uint8_t *>(wts_aie_golden.data());
  uint64_t diff_count = 0;
  for (int i = 0; i < bo_map.size(); i++) {
    auto exp = static_cast<int>(*(wts_aie_uint8_ptr + i));
    auto get = static_cast<int>(bo_map[i]);
    if (exp != get) {
      diff_count++;
    }
  }
  EXPECT_TRUE(diff_count == 0)
      << "Wts shuffle diff_count Count = " << diff_count;
}

// Helper that converts (M,K,N,Gs) to a unique string key
static inline std::string shapeToKey(int M, int K, int N, int Gs) {
  return std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(N) +
         "_" + std::to_string(Gs);
}

// Build skip set for _uint4 shapes
std::unordered_set<std::string> buildSkipSet_uint4() {
  std::unordered_set<std::string> skipSet;

  skipSet.insert(shapeToKey(1, 4096, 131072, 32));
  skipSet.insert(shapeToKey(1, 1536, 256, 32));
  skipSet.insert(shapeToKey(1, 8090, 1536, 128));
  skipSet.insert(shapeToKey(1, 8960, 1536, 32));
  skipSet.insert(shapeToKey(1, 8960, 1536, 128));
  skipSet.insert(shapeToKey(1, 9216, 2304, 32));
  skipSet.insert(shapeToKey(1, 9216, 2304, 128));
  skipSet.insert(shapeToKey(128, 4096, 32768, 128));
  skipSet.insert(shapeToKey(128, 4096, 131072, 32));
  skipSet.insert(shapeToKey(128, 4096, 131072, 128));
  skipSet.insert(shapeToKey(128, 9216, 2304, 128));
  skipSet.insert(shapeToKey(128, 9216, 2304, 32));
  skipSet.insert(shapeToKey(256, 2304, 256000, 32));
  skipSet.insert(shapeToKey(256, 4096, 128256, 128));
  skipSet.insert(shapeToKey(256, 4096, 131072, 128));
  skipSet.insert(shapeToKey(256, 4096, 131072, 32));
  skipSet.insert(shapeToKey(256, 4096, 151936, 32));
  skipSet.insert(shapeToKey(384, 2048, 128256, 128));
  skipSet.insert(shapeToKey(384, 4096, 27392, 128));
  skipSet.insert(shapeToKey(384, 4096, 65024, 128));
  skipSet.insert(shapeToKey(512, 2048, 128256, 128));
  skipSet.insert(shapeToKey(512, 2304, 256000, 128));
  skipSet.insert(shapeToKey(512, 4096, 32768, 32));
  skipSet.insert(shapeToKey(512, 4096, 32768, 128));
  skipSet.insert(shapeToKey(512, 4096, 128256, 32));
  skipSet.insert(shapeToKey(512, 4096, 128256, 128));
  skipSet.insert(shapeToKey(512, 4096, 131072, 32));
  skipSet.insert(shapeToKey(512, 4096, 131072, 128));
  skipSet.insert(shapeToKey(512, 4096, 1536000, 32));
  skipSet.insert(shapeToKey(640, 2048, 128256, 128));
  skipSet.insert(shapeToKey(640, 4096, 13696, 128));
  skipSet.insert(shapeToKey(640, 4096, 27392, 128));
  skipSet.insert(shapeToKey(640, 8192, 3072, 32));
  skipSet.insert(shapeToKey(768, 2048, 128256, 128));
  skipSet.insert(shapeToKey(768, 4096, 27392, 128));
  skipSet.insert(shapeToKey(768, 4096, 65024, 128));
  skipSet.insert(shapeToKey(896, 2048, 128256, 128));
  skipSet.insert(shapeToKey(896, 4096, 27392, 128));
  skipSet.insert(shapeToKey(1024, 2048, 128258, 128));
  skipSet.insert(shapeToKey(1024, 2304, 256000, 128));
  skipSet.insert(shapeToKey(1024, 2304, 256000, 32));
  skipSet.insert(shapeToKey(1024, 4096, 1024, 128));
  skipSet.insert(shapeToKey(1024, 4096, 14336, 128));
  skipSet.insert(shapeToKey(1024, 4096, 32768, 32));
  skipSet.insert(shapeToKey(1024, 4096, 32768, 128));
  skipSet.insert(shapeToKey(1024, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1024, 4096, 128256, 128));
  skipSet.insert(shapeToKey(1024, 4096, 129024, 128));
  skipSet.insert(shapeToKey(1024, 4096, 129024, 32));
  skipSet.insert(shapeToKey(1024, 4096, 131072, 128));
  skipSet.insert(shapeToKey(1024, 4096, 131072, 32));
  skipSet.insert(shapeToKey(1024, 4096, 151936, 128));
  skipSet.insert(shapeToKey(1024, 4096, 153600, 128));
  skipSet.insert(shapeToKey(1152, 4096, 13696, 128));
  skipSet.insert(shapeToKey(1152, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1024, 4096, 65024, 32));
  skipSet.insert(shapeToKey(1280, 2048, 8192, 128));
  skipSet.insert(shapeToKey(1280, 2048, 128256, 128));
  skipSet.insert(shapeToKey(1280, 4096, 27392, 128));
  skipSet.insert(shapeToKey(1280, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1408, 2048, 8192, 128));
  skipSet.insert(shapeToKey(1408, 2048, 128256, 128));
  skipSet.insert(shapeToKey(1408, 4096, 65024, 32));
  skipSet.insert(shapeToKey(1536, 3072, 128256, 128));
  skipSet.insert(shapeToKey(1536, 2048, 3072, 128));
  skipSet.insert(shapeToKey(1536, 4096, 4608, 128));
  skipSet.insert(shapeToKey(1536, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1664, 2048, 128256, 128));
  skipSet.insert(shapeToKey(1664, 4096, 13696, 128));
  skipSet.insert(shapeToKey(1664, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1664, 8192, 3072, 128));
  skipSet.insert(shapeToKey(1792, 2048, 128256, 128));
  skipSet.insert(shapeToKey(1792, 4096, 4608, 128));
  skipSet.insert(shapeToKey(1792, 4096, 27392, 128));
  skipSet.insert(shapeToKey(1792, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1920, 2048, 128256, 128));
  skipSet.insert(shapeToKey(1920, 4096, 27392, 128));
  skipSet.insert(shapeToKey(1920, 4096, 65024, 32));
  skipSet.insert(shapeToKey(1920, 4096, 65024, 128));
  skipSet.insert(shapeToKey(2048, 2048, 128256, 128));
  skipSet.insert(shapeToKey(2048, 2304, 9216, 128));
  skipSet.insert(shapeToKey(2048, 2304, 256000, 128));
  skipSet.insert(shapeToKey(2048, 3072, 128256, 128));
  skipSet.insert(shapeToKey(2048, 4096, 12288, 32));
  skipSet.insert(shapeToKey(2048, 4096, 22016, 128));
  skipSet.insert(shapeToKey(2048, 4096, 22528, 128));
  skipSet.insert(shapeToKey(2048, 4096, 27392, 128));
  skipSet.insert(shapeToKey(2048, 4096, 32768, 128));
  skipSet.insert(shapeToKey(2048, 4096, 65024, 128));
  skipSet.insert(shapeToKey(2048, 4096, 128256, 32));
  skipSet.insert(shapeToKey(2048, 4096, 128256, 128));
  skipSet.insert(shapeToKey(2048, 4096, 129024, 128));
  skipSet.insert(shapeToKey(2048, 4096, 129024, 32));
  skipSet.insert(shapeToKey(2048, 4096, 131072, 32));
  skipSet.insert(shapeToKey(2048, 4096, 131072, 128));
  skipSet.insert(shapeToKey(2048, 4096, 151936, 32));
  skipSet.insert(shapeToKey(2048, 4096, 151936, 128));
  skipSet.insert(shapeToKey(2048, 4096, 153600, 128));
  skipSet.insert(shapeToKey(2048, 11008, 4096, 128));
  skipSet.insert(shapeToKey(2048, 11008, 4096, 32));
  skipSet.insert(shapeToKey(3072, 2048, 128256, 128));
  skipSet.insert(shapeToKey(3072, 4096, 4608, 128));
  skipSet.insert(shapeToKey(3072, 4096, 27392, 128));
  skipSet.insert(shapeToKey(3072, 4096, 65024, 128));
  skipSet.insert(shapeToKey(1, 8960, 1536, 32));
  skipSet.insert(shapeToKey(1, 8960, 1536, 128));
  skipSet.insert(shapeToKey(1, 9216, 2304, 32));
  skipSet.insert(shapeToKey(1, 9216, 2304, 128));
  skipSet.insert(shapeToKey(1, 18944, 3584, 128));
  skipSet.insert(shapeToKey(128, 9216, 2304, 128));
  skipSet.insert(shapeToKey(2048, 3072, 128256, 128));
  skipSet.insert(shapeToKey(3072, 3072, 128256, 128));

  return skipSet;
}

// Build a skip set for _int4 test shapes
std::unordered_set<std::string> buildSkipSet_int4() {
  std::unordered_set<std::string> skipSet;

  skipSet.insert(shapeToKey(1, 8960, 1536, 32));
  skipSet.insert(shapeToKey(1, 8960, 1536, 128));
  skipSet.insert(shapeToKey(1, 9216, 2304, 32));
  skipSet.insert(shapeToKey(1, 9216, 2304, 128));
  skipSet.insert(shapeToKey(1, 18944, 3584, 128));
  skipSet.insert(shapeToKey(128, 9216, 2304, 32));
  skipSet.insert(shapeToKey(128, 9216, 2304, 128));
  skipSet.insert(shapeToKey(1536, 3072, 128256, 128));
  skipSet.insert(shapeToKey(2048, 3072, 128256, 128));
  skipSet.insert(shapeToKey(3072, 3072, 128256, 128));

  return skipSet;
}

TEST(Qlinear_2Testw3a16_int4, AutoRunAllTxnShapes) {
  // 1. Create an operator that can parse transaction files and fill
  // supported_shapes_.
  using MladfOp = ryzenai::mladfmatmulbias<int16_t, int8_t, int16_t, int16_t>;
  MladfOp shapeFinderOp(
      /*a_dtype=*/"bfloat16",
      /*b_dtype=*/"int4",
      /*c_dtype=*/"bfloat16",
      /*load_xrt=*/true,
      /*attr=*/std::map<std::string, std::any>());

  // 3. Retrieve the discovered shapes
  auto shapes = std::vector(shapeFinderOp.get_supported_shapes());

  auto skipSet = buildSkipSet_int4();

  // Remove shapes if they're in the skipSet
  shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                              [&](const auto &s) {
                                std::string key =
                                    shapeToKey(s.M, s.K, s.N, s.Gs);
                                return skipSet.find(key) != skipSet.end();
                              }),
               shapes.end());

  // 4. Loop over each shape, call test_matmul_mladf(...)
  for (auto &s : shapes) {
    int err_count =
        test_matmul_mladf<int16_t, int8_t, int16_t>(s.M, s.K, s.N,
                                                    /*debug=*/false,
                                                    /*a_dtype=*/"bfloat16",
                                                    /*b_dtype=*/"int4",
                                                    /*c_dtype=*/"bfloat16",
                                                    s.Gs, // group_size
                                                    /*compare_values=*/true,
                                                    /*op_version=*/"v1");

    EXPECT_EQ(err_count, 0) << "[test_matmul_mladf] error count=" << err_count
                            << " for shape M=" << s.M << ", K=" << s.K
                            << ", N=" << s.N << ", Gs=" << s.Gs;
  }
}

TEST(Qlinear_2Testw3a16_uint4, AutoRunAllTxnShapes) {
  // 1. Create an operator that can parse transaction files and fill
  // supported_shapes_.
  using MladfOp = ryzenai::mladfmatmulbias<int16_t, int8_t, int16_t, int16_t>;
  MladfOp shapeFinderOp(
      /*a_dtype=*/"bfloat16",
      /*b_dtype=*/"uint4",
      /*c_dtype=*/"bfloat16",
      /*load_xrt=*/true,
      /*attr=*/std::map<std::string, std::any>());

  // 3. Retrieve the discovered shapes
  auto shapes = std::vector(shapeFinderOp.get_supported_shapes());

  auto skipSet = buildSkipSet_uint4();

  // Remove shapes if they're in the skipSet
  shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                              [&](const auto &s) {
                                std::string key =
                                    shapeToKey(s.M, s.K, s.N, s.Gs);
                                return skipSet.find(key) != skipSet.end();
                              }),
               shapes.end());

  // 4. Loop over each shape, call test_matmul_mladf(...)
  for (auto &s : shapes) {
    int err_count =
        test_matmul_mladf<int16_t, int8_t, int16_t>(s.M, s.K, s.N,
                                                    /*debug=*/false,
                                                    /*a_dtype=*/"bfloat16",
                                                    /*b_dtype=*/"int4",
                                                    /*c_dtype=*/"bfloat16",
                                                    s.Gs, // group_size
                                                    /*compare_values=*/true,
                                                    /*op_version=*/"v1");

    EXPECT_EQ(err_count, 0) << "[test_matmul_mladf] error count=" << err_count
                            << " for shape M=" << s.M << ", K=" << s.K
                            << ", N=" << s.N << ", Gs=" << s.Gs;
  }
}

TEST(Bfp16Gemm, ChatGLM_Kernel) {
  std::vector<ryzenai::mladf_matrix_shapes> shapes = {
      {2048, 4096, 4096, 128},
      {2048, 4096, 4608, 128},
      {2048, 4096, 65024, 128},
      {2048, 13696, 4096, 128},
      // {2048, 4096, 13696, 128}
  };

  for (auto &s : shapes) {
    int err_count = test_bfp16_cast_gemm_mladf<uint16_t, uint8_t, uint16_t>(
        s.M, s.K, s.N,
        /*debug=*/false,
        /*a_dtype=*/"bfloat16",
        /*b_dtype=*/"uint4",
        /*c_dtype=*/"bfloat16",
        /*group size=*/s.Gs,
        /*compare_values=*/true,
        /*op_version=*/"bfp16gemm");

    EXPECT_EQ(err_count, 0) << "[test_matmul_mladf] error count=" << err_count
                            << " for shape M=" << s.M << ", K=" << s.K
                            << ", N=" << s.N << ", Gs=" << s.Gs;
  }
}

TEST(Qlinear_2Testw3a16_bfp16, ChatGLM_Kernel) {
  std::vector<ryzenai::mladf_matrix_shapes> shapes = {
      {2048, 4096, 4096, 128},  {2048, 4096, 4608, 128},
      {2048, 13696, 4096, 128}, {2048, 4096, 13696, 128},
      {2048, 4096, 65024, 128}, {2048, 4096, 3900, 128}};

  for (auto &s : shapes) {
    int err_count = test_matmul_mladf<int16_t, uint8_t, int16_t>(
        s.M, s.K, s.N,
        /*debug=*/false,
        /*a_dtype=*/"bfloat16",
        /*b_dtype=*/"uint4",
        /*c_dtype=*/"bfloat16",
        /*group size=*/s.Gs,
        /*compare_values=*/true,
        /*op_version=*/"bfp16gemm");

    EXPECT_EQ(err_count, 0) << "[test_matmul_mladf] error count=" << err_count
                            << " for shape M=" << s.M << ", K=" << s.K
                            << ", N=" << s.N << ", Gs=" << s.Gs;
  }
}

// M = 1, updated overlay
// TEST(Bfp16Gemm, ChatGLM_Kernel_Golden) {
//   std::vector<ryzenai::mladf_matrix_shapes> shapes = {
//       {2048, 4096, 4096, 128},  {2048, 4096, 4608, 128},
//       {2048, 13696, 4096, 128}, {2048, 4096, 13696, 128},
//       {2048, 4096, 65024, 128}};

//   for (auto &s : shapes) {
//     int err_count = test_bfp16_cast_gemm_mladf<int16_t, uint8_t, int16_t>(
//         s.M, s.K, s.N,
//         /*debug=*/true,
//         /*a_dtype=*/"bfloat16",
//         /*b_dtype=*/"uint4",
//         /*c_dtype=*/"bfloat16",
//         /*group size=*/s.Gs,
//         /*compare_values=*/true,
//         /*op_version=*/"v2");

//     EXPECT_EQ(err_count, 0) << "[test_bfp16_cast_gemm_mladf] error count=" <<
//     err_count
//                             << " for shape M=" << s.M << ", K=" << s.K
//                             << ", N=" << s.N << ", Gs=" << s.Gs;
//   }
// }

TEST(Qlinear_2Testw3a16_bfp16, new_bfp16_random) {
  std::vector<ryzenai::mladf_matrix_shapes> shapes = {
      {2048, 4096, 4096, 128},  {2048, 4096, 4096, 32},
      {2048, 4096, 4608, 128},  {2048, 4096, 4608, 32},
      {2048, 13696, 4096, 128}, {2048, 13696, 4096, 32},
      {2048, 4096, 13696, 128}, {2048, 4096, 13696, 32},
      {2048, 4096, 65024, 128}, {2048, 4096, 3900, 32}};

  for (auto &s : shapes) {
    int err_count =
        test_matmul_mladf<int16_t, uint8_t, int16_t>(s.M, s.K, s.N,
                                                     /*debug=*/false,
                                                     /*a_dtype=*/"bfloat16",
                                                     /*b_dtype=*/"uint4",
                                                     /*c_dtype=*/"bfloat16",
                                                     /*group size=*/s.Gs,
                                                     /*compare_values=*/true,
                                                     /*op_version=*/"v2");

    EXPECT_EQ(err_count, 0) << "[test_matmul_mladf] error count=" << err_count
                            << " for shape M=" << s.M << ", K=" << s.K
                            << ", N=" << s.N << ", Gs=" << s.Gs;
  }
}
// ------------- TEST END for mladfmatmulbias -------------

// Deepseek
// 7b
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1x3584x4608_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 3584, 4608, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1x3584x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 3584, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1x3584x18944_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 3584, 18944, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1x18944x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 18944, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_128x3584x4608_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 3584, 4608, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_128x3584x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 3584, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_128x3584x18944_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 3584, 18944, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_128x18944x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 18944, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_256x3584x4608_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 3584, 4608, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_256x3584x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 3584, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_256x3584x18944_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 3584, 18944, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_256x18944x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 18944, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_512x3584x4608_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 3584, 4608, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_512x3584x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 3584, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_512x3584x18944_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 3584, 18944, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_512x18944x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 18944, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1024x3584x4608_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 3584, 4608, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1024x3584x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 3584, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1024x3584x18944_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 3584, 18944, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_1024x18944x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 18944, 3584, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_2048x3584x4608_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 3584, 4608, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_2048x3584x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 3584, 3584, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_2048x3584x18944_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 3584, 18944, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen7b_2Testw3a16_high_time, Kernel4mladf_2048x18944x3584_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 18944, 3584, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 1.5b

TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_1x1536x2048_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 1536, 2048, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_1x1536x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 1536, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_1x1536x8960_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 1536, 8960, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_1x8960x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1, 8960, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_128x1536x2048_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 1536, 2048, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_128x1536x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 1536, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_128x1536x8960_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 1536, 8960, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_128x8960x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      128, 8960, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_256x1536x2048_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 1536, 2048, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_256x1536x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 1536, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_256x1536x8960_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 1536, 8960, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_256x8960x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      256, 8960, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_512x1536x2048_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 1536, 2048, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_512x1536x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 1536, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_512x1536x8960_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 1536, 8960, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time, Kernel4mladf_512x8960x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      512, 8960, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_1024x1536x2048_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 1536, 2048, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_1024x1536x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 1536, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_1024x1536x8960_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 1536, 8960, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_1024x8960x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      1024, 8960, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_2048x1536x2048_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 1536, 2048, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_2048x1536x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 1536, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_2048x1536x8960_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 1536, 8960, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qwen1_5b_2Testw3a16_high_time,
     Kernel4mladf_2048x8960x1536_int4_grp128_v1) {
  int err_count = test_matmul_mladf<uint16_t, int8_t, uint16_t>(
      2048, 8960, 1536, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
