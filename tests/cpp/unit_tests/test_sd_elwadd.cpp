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
#include <ops/sd/elwadd.hpp>

// #pragma STDC FENV_ACCESS ON
static float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

static int get_shape_ele_num(const std::vector<int> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
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
int sd_elwadd_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

static void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_sd_elwadd(const std::vector<int> &a_shape,
                   const std::vector<int> &b_shape, bool debug = false,
                   const std::string &a_type = "bfloat16", // a bo
                   const std::string &b_type = "bfloat16", // b bo
                   const std::string &c_type = "bfloat16", // c bo
                   const std::string &model_name = "SD_VAE_DEC",
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
  std::vector<int> c_shape;
  if (a_shape.size() == b_shape.size()) {
    int a_size = get_shape_ele_num(a_shape);
    int b_size = get_shape_ele_num(b_shape);
    c_shape = a_size > b_size ? a_shape : b_shape;
  } else {
    if (a_shape.size() > b_shape.size()) {
      c_shape = a_shape;
    } else {
      c_shape = b_shape;
    }
  }

  std::vector<OuT> aie_out(get_shape_ele_num(c_shape));
  std::map<std::string, std::any> attr;
  attr["a_shape"] = a_shape;
  attr["b_shape"] = b_shape;
  attr["c_shape"] = c_shape;

  std::vector<size_t> a_size_t_shape;
  std::transform(a_shape.begin(), a_shape.end(),
                 std::back_inserter(a_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::vector<size_t> b_size_t_shape;
  std::transform(b_shape.begin(), b_shape.end(),
                 std::back_inserter(b_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::vector<size_t> c_size_t_shape;
  std::transform(c_shape.begin(), c_shape.end(),
                 std::back_inserter(c_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::string shape_key;
  if (test_with_golden) {
    ryzenai::sd::elwadd sd_elwadd =
        ryzenai::sd::elwadd<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_type, b_type, c_type, false, attr);
    sd_elwadd.debug(debug);
    sd_elwadd.set_params(model_name, a_shape, b_shape);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_elwadd/";
    shape_key = txnbin_a_header.at(a_type) + txnbin_b_header.at(b_type) +
                txnbin_acc_header.at(c_type);
    for (int i = 0; i < a_shape.size(); i++) {
      shape_key += "_" + std::to_string(a_shape[i]);
    }
    shape_key += "__";
    for (int i = 0; i < b_shape.size(); i++) {
      shape_key += "_" + std::to_string(b_shape[i]);
    }
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> b_aie = read_hex_file(wts_path);
    std::vector<Tensor> const_Tensor;
    std::vector<Tensor> input_Tensor;
    input_Tensor.push_back({a_aie.data(), a_size_t_shape, a_type});

    if (sd_elwadd.is_bias_cal()) {
      const_Tensor.push_back({b_aie.data(), b_size_t_shape, b_type});
    } else {
      input_Tensor.push_back({b_aie.data(), b_size_t_shape, b_type});
    }
    sd_elwadd.initialize_const_params(const_Tensor);

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_size_t_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(sd_elwadd.execute(input_Tensor, output_Tensor));
#else
    sd_elwadd.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());

    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    quantize_err_count = sd_elwadd_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::elwadd sd_elwadd =
        ryzenai::sd::elwadd<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_type, b_type, c_type, false, attr);
    sd_elwadd.debug(debug);
    sd_elwadd.set_params(model_name, a_shape, b_shape);
    // gen rand
    std::vector<float> raw_a(get_shape_ele_num(a_shape), 0);
    initialize_random_float(raw_a, 2, -2);
    auto bf16_a = float_2_bf16_vec(raw_a);

    std::vector<float> raw_b(get_shape_ele_num(b_shape), 0);
    initialize_random_float(raw_b, 2, -2);
    auto bf16_b = float_2_bf16_vec(raw_b);

    std::vector<Tensor> const_Tensor;
    std::vector<Tensor> input_Tensor;

    input_Tensor.push_back({bf16_a.data(), a_size_t_shape, a_type});

    if (sd_elwadd.is_bias_cal()) {
      const_Tensor.push_back({bf16_b.data(), b_size_t_shape, b_type});
    } else {
      input_Tensor.push_back({bf16_b.data(), b_size_t_shape, b_type});
    }
    sd_elwadd.initialize_const_params(const_Tensor);

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_size_t_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(sd_elwadd.execute(input_Tensor, output_Tensor));
#else
    sd_elwadd.execute(input_Tensor, output_Tensor);
#endif
    std::vector<int64_t> a_int64_shape;
    for (int val : a_shape) {
      a_int64_shape.push_back(static_cast<int64_t>(val));
    }

    std::vector<int64_t> b_int64_shape;
    for (int val : b_shape) {
      b_int64_shape.push_back(static_cast<int64_t>(val));
    }
    auto tensor_a =
        torch::from_blob(raw_a.data(), a_int64_shape, torch::kFloat);
    auto tensor_b =
        torch::from_blob(raw_b.data(), b_int64_shape, torch::kFloat);
    auto tensor_c = tensor_a + tensor_b;

    std::vector<float> torch_c_buffer(tensor_c.numel());
    std::memcpy(torch_c_buffer.data(), tensor_c.data_ptr<float>(),
                tensor_c.numel() * sizeof(float));
    auto torch_c_bf16 = float_2_bf16_vec(torch_c_buffer);
    quantize_err_count = sd_elwadd_check_result<OuT>(
        torch_c_bf16, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return quantize_err_count;
}

// Golden unittest
// Unet
TEST(SD_ELWADD_Test, Golden_UnetAddLayer1) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 64, 320}, {2, 1, 1, 320}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer2) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 1, 1, 640}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer3) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 1, 1, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer4) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 1, 1, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer5) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 64, 320}, {2, 64, 64, 320}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer6) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 32, 32, 640}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer7) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 16, 16, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer8) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 8, 8, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer9) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 320}, {320}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer10) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 640}, {640}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer11) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer12) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer13) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 2560}, {2560}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer14) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 5120}, {5120}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer15) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 10240}, {10240}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer16) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 10240}, {10240}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer17) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 320}, {2, 4096, 320}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer18) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 640}, {2, 1024, 640}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer19) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 1280}, {2, 256, 1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer20) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 1280}, {2, 64, 1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer21) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer22) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 640}, {640}, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer23) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 320}, {320}, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer24) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer25) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 2560}, {2560}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer26) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 5120}, {5120}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_UnetAddLayer27) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 5120}, {5120}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// VAE Decoder

TEST(SD_ELWADD_Test, Golden_VaeAddLayer1) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 64, 64, 512}, {1, 64, 64, 512}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_VaeAddLayer2) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 128, 128, 512}, {1, 128, 128, 512}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_VaeAddLayer3) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 256, 256, 256}, {1, 256, 256, 256}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_VaeAddLayer4) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 512, 512, 128}, {1, 512, 512, 128}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_VaeAddLayer5) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 4096, 512}, {512}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD3

TEST(SD_ELWADD_Test, Golden_SD3AddLayer1) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 1024, 1024, 128}, {1, 1024, 1024, 128}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer2) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 16384, 512}, {512}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer3) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 256, 256, 512}, {1, 256, 256, 512}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer4) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 512, 512, 256}, {1, 512, 512, 256}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer5) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer6) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {1, 1024, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer7) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer8) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer9) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 1024, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer10) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer11) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 64}, {64}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer12) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer13) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1536}, {2, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer14) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer15) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer16) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 1536}, {2, 154, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer17) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer18) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer19) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer20) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer21) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {1, 4096, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer22) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer23) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer24) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 4096, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer25) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer26) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 64}, {64}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Golden_SD3AddLayer27) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 1536}, {2, 333, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Random unittest
// Unet
TEST(SD_ELWADD_Test, Random_UnetAddLayer1) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 64, 320}, {2, 1, 1, 320}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer2) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 1, 1, 640}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer3) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 1, 1, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer4) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 1, 1, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer5) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 64, 320}, {2, 64, 64, 320}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer6) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 32, 32, 640}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer7) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 16, 16, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer8) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 8, 8, 1280}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer9) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 320}, {320}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer10) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 640}, {640}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer11) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer12) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer13) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 2560}, {2560}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer14) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 5120}, {5120}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer15) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 10240}, {10240}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer16) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 10240}, {10240}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer17) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 320}, {2, 4096, 320}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer18) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 640}, {2, 1024, 640}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer19) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 1280}, {2, 256, 1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer20) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 1280}, {2, 64, 1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer21) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer22) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 640}, {640}, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer23) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 320}, {320}, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer24) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1280}, {1280}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer25) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 2560}, {2560}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer26) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 256, 5120}, {5120}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_UnetAddLayer27) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 64, 5120}, {5120}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// VAE Decoder

TEST(SD_ELWADD_Test, Random_VaeAddLayer1) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 64, 64, 512}, {1, 64, 64, 512}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_VaeAddLayer2) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 128, 128, 512}, {1, 128, 128, 512}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_VaeAddLayer3) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 256, 256, 256}, {1, 256, 256, 256}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_VaeAddLayer4) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 512, 512, 128}, {1, 512, 512, 128}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_VaeAddLayer5) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 4096, 512}, {512}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD3

TEST(SD_ELWADD_Test, Random_SD3AddLayer1) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 1024, 1024, 128}, {1, 1024, 1024, 128}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer2) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 16384, 512}, {512}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer3) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 256, 256, 512}, {1, 256, 256, 512}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer4) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {1, 512, 512, 256}, {1, 512, 512, 256}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer5) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer6) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {1, 1024, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer7) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer8) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer9) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 1024, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer10) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer11) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1024, 64}, {64}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer12) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer13) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 1536}, {2, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer14) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer15) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer16) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 1536}, {2, 154, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer17) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 154, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer18) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer19) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer20) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer21) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {1, 4096, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer22) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer23) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 1, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer24) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 4096, 1536}, false, "bfloat16", "bfloat16",
      "bfloat16", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer25) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 6144}, {6144}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer26) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 4096, 64}, {64}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_ELWADD_Test, Random_SD3AddLayer27) {
  int err_count = test_sd_elwadd<uint16_t, uint16_t, uint16_t>(
      {2, 333, 1536}, {2, 333, 1536}, false, "bfloat16", "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
