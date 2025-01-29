/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <ops/sd/resize.hpp>

// #pragma STDC FENV_ACCESS ON
static int test_count = 0;

static int get_shape_ele_num(const std::vector<int> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

static float bfloat16_to_float(uint16_t x) {
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

static void dumpVectorToTxt(const std::vector<float> &data,
                            const std::vector<int> &c_shape,
                            const std::string &filename) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  int N = c_shape[0];
  int H = c_shape[1];
  int W = c_shape[2];
  int C = c_shape[3];
  int index = 0;

  for (int n = 0; n < N; ++n) {
    outFile << "Batch " << n << ":\n";
    for (int h = 0; h < H; ++h) {
      outFile << "  Height " << h << ":\n";
      for (int w = 0; w < W; ++w) {
        outFile << "    Width " << w << ": ";
        for (int c = 0; c < C; ++c) {
          outFile << std::setw(8) << data[index++] << " ";
        }
        outFile << "\n";
      }
      outFile << "\n";
    }
    outFile << "\n";
  }

  outFile.close();
  std::cout << "Data dumped to " << filename << std::endl;
}

template <typename T>
int sd_resize_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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
      if (err_count < 20) {
        std::cout << "ERROR: Y[" << i << "]: "
                  << "Expected: " << bfloat16_to_float(cpu_Y.at(i)) << ","
                  << "Received: " << bfloat16_to_float(aie_Y.at(i)) << "\n";
      }
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

static double round_half_2_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

static void aie_srs_compute(std::vector<uint32_t> &input_output) {
  int data_width = 16;
  int shift = 16;
  for (size_t i = 0; i < input_output.size(); ++i) {
    double temp = static_cast<double>(input_output[i]) / std::pow(2.0, shift);
    // temp = std::round(temp);
    temp = round_half_2_even(temp);
    if (temp > std::pow(2.0f, data_width) - 1) {
      temp = float(std::pow(2.0f, data_width)) - 1;
    }
    if (temp < 0) {
      temp = 0;
    }
    input_output[i] = static_cast<uint32_t>(temp);
  }
}

static std::vector<uint16_t> float_2_bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs_compute(x_uint32);

  std::vector<uint16_t> x_uint16(x.size());

  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint16[i] = static_cast<uint16_t>(x_uint32[i]);
  }

  return x_uint16;
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
static int get_exponent_cpu(float v) {
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
static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

static void dumpToBinFile(const std::vector<uint16_t> &data,
                          const std::string &filename) {
  std::ofstream outFile(filename, std::ios::out | std::ios::binary);
  if (!outFile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < data.size(); i += 2) {
    outFile << std::hex << data[i + 1] << data[i];
    if (i % 2 == 0) {
      outFile << "\n";
    }
  }
  outFile.close();
  std::cout << "Data dumped to bin file " << filename << std::endl;
}

template <typename InT, typename OuT>
int test_sd_resize(const std::vector<int> &a_shape,
                   const std::vector<int> &c_shape, bool debug = false,
                   const std::string &a_type = "bfloat16", // a bo
                   const std::string &c_type = "bfloat16", // c bo
                   const std::string &model_name = "SD_VAE_DEC",
                   float pixel_L2_norm_tolerance = 0.01,
                   bool test_with_golden = false) {
  int quantize_err_count = 0;
  int unquantize_err_count = 0;
  float error_tolerance = 0.01;
  std::map<std::string, std::string> txnbin_a_header = {
      {"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  std::map<std::string, std::string> txnbin_c_header = {
      {"bfloat16", "acc16bf"}};
  std::vector<OuT> aie_out(get_shape_ele_num(c_shape));
  std::map<std::string, std::any> attr;
  attr["a_shape"] = a_shape;
  attr["c_shape"] = c_shape;

  std::vector<size_t> a_size_t_shape;
  std::transform(a_shape.begin(), a_shape.end(),
                 std::back_inserter(a_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::vector<size_t> c_size_t_shape;
  std::transform(c_shape.begin(), c_shape.end(),
                 std::back_inserter(c_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::string shape_key =
      txnbin_a_header.at(a_type) + txnbin_c_header.at(c_type);
  for (int i = 0; i < a_shape.size(); i++) {
    shape_key += "_" + std::to_string(a_shape[i]);
  }
  shape_key += "__";
  for (int i = 0; i < c_shape.size(); i++) {
    shape_key += "_" + std::to_string(c_shape[i]);
  }
  if (test_with_golden) {
    ryzenai::sd::resize sd_resize =
        ryzenai::sd::resize<std::uint16_t, std::uint16_t>(a_type, c_type, false,
                                                          attr);
    sd_resize.debug(debug);
    sd_resize.set_params(model_name, a_shape, c_shape);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_resize/";
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);
    std::vector<float> golden_ifm_float(get_shape_ele_num(a_shape));
    std::vector<OuT> golden_ifm_bf16(get_shape_ele_num(a_shape));
    // memcpy(golden_ifm_bf16.data(), a_aie.data(), a_aie.size() *
    // sizeof(uint32_t)); for (int i =0;i<golden_ifm_bf16.size(); ++i) {
    //   golden_ifm_float[i] = bfloat16_to_float(golden_ifm_bf16[i]);
    // }
    // dumpVectorToTxt(golden_ifm_float, a_shape, "gt_aie_input.txt");

    std::vector<Tensor> const_Tensor;
    sd_resize.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;
    input_Tensor.push_back({a_aie.data(), a_size_t_shape, a_type});

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_size_t_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(sd_resize.execute(input_Tensor, output_Tensor));
#else
    sd_resize.execute(input_Tensor, output_Tensor);
#endif
    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));

    if (test_count) {
      // dumpToBinFile(bf16_output_golden, "gt_golden.bin");
      // dumpToBinFile(aie_out, "gt_aie_out.bin");
      // std::vector<float> aie_out_flow(aie_out.size(), 0);
      // for (int i = 0; i < aie_out.size(); ++i) {
      //   aie_out_flow[i] = bfloat16_to_float(aie_out[i]);
      // }
      // dumpVectorToTxt(aie_out_flow, c_shape, "gt_aie_output.txt");

      // std::vector<float> f_output_golden(bf16_output_golden.size(), 0);
      // for (int i =0;i<aie_out.size(); ++i) {
      //   f_output_golden[i] = bfloat16_to_float(bf16_output_golden[i]);
      // }
      // dumpVectorToTxt(f_output_golden, c_shape, "gt_golden.txt");
    }

    quantize_err_count = sd_resize_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::resize sd_resize =
        ryzenai::sd::resize<std::uint16_t, std::uint16_t>(a_type, c_type, false,
                                                          attr);
    sd_resize.debug(debug);
    sd_resize.set_params(model_name, a_shape, c_shape);
    // gen rand
    std::vector<float> raw_a(get_shape_ele_num(a_shape), 0);
    initialize_random_float(raw_a, 2, -2);
    // dumpVectorToTxt(raw_a, a_shape, "random_aie_input.txt");
    auto bf16_a = float_2_bf16_vec(raw_a);

    std::vector<Tensor> const_Tensor;
    sd_resize.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{bf16_a.data(), a_size_t_shape, a_type}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_size_t_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(sd_resize.execute(input_Tensor, output_Tensor));
#else
    sd_resize.execute(input_Tensor, output_Tensor);
#endif
    // std::vector<float> aie_out_flow(aie_out.size());
    // for (int i = 0; i < aie_out.size(); ++i) {
    //   aie_out_flow[i] = bfloat16_to_float(aie_out[i]);
    // }
    // dumpVectorToTxt(aie_out_flow, c_shape, "random_aie_output.txt");

    std::vector<int64_t> a_int64_shape;
    for (int val : a_shape) {
      a_int64_shape.push_back(static_cast<int64_t>(val));
    }
    std::vector<int64_t> c_int64_shape = {c_shape[1], c_shape[2]};
    at::Tensor tensor_a =
        torch::from_blob(raw_a.data(), a_int64_shape, torch::kFloat);

    torch::Tensor tensor_a_nchw = tensor_a.permute({0, 3, 1, 2});
    tensor_a_nchw = tensor_a_nchw.contiguous();

    at::Tensor tensor_c = torch::nn::functional::interpolate(
        tensor_a_nchw, torch::nn::functional::InterpolateFuncOptions()
                           .size(c_int64_shape)
                           .mode(torch::kNearest));
    tensor_c = tensor_c.permute({0, 2, 3, 1}).contiguous();

    std::vector<float> torch_output_buffer(tensor_c.numel());
    std::memcpy(torch_output_buffer.data(), tensor_c.data_ptr<float>(),
                tensor_c.numel() * sizeof(float));
    // dumpVectorToTxt(torch_output_buffer, c_shape, "random_torch_output.txt");

    auto torch_output_bf16 = float_2_bf16_vec(torch_output_buffer);
    quantize_err_count = sd_resize_check_result<OuT>(
        torch_output_bf16, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  test_count++;
  return quantize_err_count;
}

// Golden unittest
// TEST(SD_RESIZE_Test, Golden_Layer1) {
//   int err_count = test_sd_resize<uint16_t, uint16_t>(
//       {1, 64, 64, 512}, {1, 128, 128, 512}, false, "bfloat16", "bfloat16",
//       "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

TEST(SD_RESIZE_Test, Golden_Layer2) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 128, 128, 512}, {1, 256, 256, 512}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Golden_Layer3) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 256, 256, 256}, {1, 512, 512, 256}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// TEST(SD_RESIZE_Test, Golden_Layer4) {
//   int err_count = test_sd_resize<uint16_t, uint16_t>(
//       {2, 16, 16, 1280}, {2, 32, 32, 1280}, false, "bfloat16", "bfloat16",
//       "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// TEST(SD_RESIZE_Test, Golden_Layer5) {
//   int err_count = test_sd_resize<uint16_t, uint16_t>(
//       {2, 32, 32, 640}, {2, 64, 64, 640}, false, "bfloat16", "bfloat16",
//       "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// TEST(SD_RESIZE_Test, Golden_Layer6) {
//   int err_count = test_sd_resize<uint16_t, uint16_t>(
//       {2, 8, 8, 1280}, {2, 16, 16, 1280}, false, "bfloat16", "bfloat16",
//       "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// SD3
TEST(SD_RESIZE_Test, Golden_SD3Layer1) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 256, 256, 512}, {1, 512, 512, 512}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Golden_SD3Layer2) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 512, 512, 256}, {1, 1024, 1024, 256}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Random unittest
TEST(SD_RESIZE_Test, Random_Layer1) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 64, 64, 512}, {1, 128, 128, 512}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Random_Layer2) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 128, 128, 512}, {1, 256, 256, 512}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Random_Layer3) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 256, 256, 256}, {1, 512, 512, 256}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Random_Layer4) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 32, 32, 1280}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Random_Layer5) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 64, 64, 640}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Random_Layer6) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 16, 16, 1280}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD3
TEST(SD_RESIZE_Test, Random_SD3Layer1) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 256, 256, 512}, {1, 512, 512, 512}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_RESIZE_Test, Random_SD3Layer2) {
  int err_count = test_sd_resize<uint16_t, uint16_t>(
      {1, 512, 512, 256}, {1, 1024, 1024, 256}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
