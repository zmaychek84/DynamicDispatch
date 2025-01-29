/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/lstm/lstm.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

using namespace std;

struct scs_zps {
  std::vector<float> scales;
  std::vector<int> zps;
};
std::map<std::string, struct scs_zps> scales_zps = {
    {"320",
     {{0.00010169961751671508, 0.00008803531090961769, 0.00009158127795672044,
       0.00005283662903821096, 0.00003045230005227495, 0.00003045230005227495,
       0.00010079697676701471, 0.00008443013211945072, 0.00003503978223307058,
       0.000030487684853142127},
      {25376, 26093, 36514, 31887, 32806, 32806, 28437, 33184, 25612, 32771}}},
    {"640",
     {{0.00010395312710897997, 0.00008803531090961769, 0.00009158127795672044,
       0.00005283662903821096, 0.00003044728327949997, 0.00003044728327949997,
       0.00010079697676701471, 0.00008443013211945072, 0.00003503978223307058,
       0.000030487981348414905},
      {26247, 26093, 36514, 31887, 32810, 32810, 28437, 33184, 25612, 32761}}},
    {"1280",
     {{0.00010179720266023651, 0.00008803531090961769, 0.00009158127795672044,
       0.00005283662903821096, 0.000030468165277852677, 0.000030468165277852677,
       0.00010079697676701471, 0.00008443013211945072, 0.00003503978223307058,
       0.00003049135921173729},
      {25415, 26093, 36514, 31887, 32788, 32788, 28437, 33184, 25612, 32760}}}
    /*{"2560", {{}, {}}} */};

static std::string GetTestSubFolderName(std::string prefix, int Mi0, int Mi1,
                                        int Mi2) {
  return prefix + "_" + std::to_string(Mi0) + "_" + std::to_string(Mi1) + "_" +
         std::to_string(Mi2);
}

static int CompareFileContents(const std::string &input_file_path,
                               const std::string &output_file_path) {
  // Open the input file and read the contents into a string
  std::ifstream input_file(input_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_file_path << std::endl;
    return -1;
  }

  std::ifstream temp1(input_file_path);
  std::ifstream temp2(output_file_path);
  uint32_t val1, val2;
  int errcount = 0;
  while (temp1 >> val1 && temp2 >> val2) {
    if (val1 != val2) {
      std::cout << "ERROR: Expected: " << val1 << ", "
                << "Received: " << val2 << "\n";
      std::cout << val1 << " " << val2 << std::endl;
      errcount++;
    }
  }
  std::cout << "Errcount : " << errcount << std::endl;
  return errcount;
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_lstm(int Mi0, int Mi1, int Mi2, int Mo0, int Mo1, int Mo2,
              bool debug = false, const std::string &ifmDtype = "uint16",
              const std::string &weightDtype = "uint16",
              const std::string &ofmDtype = "uint16",
              const int modelNum = 320) {

  int err_count = 0;
  std::string fileName, testDataFolder, generatedFileName;

  testDataFolder = OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" +
                   "cpp" + "\\" + "unit_tests" + "\\" + "testDataMladf" + "\\" +
                   "lstm" + "_" + std::to_string(modelNum);

  std::vector<size_t> lstm0_w_weightShape = {2, 512, static_cast<size_t>(Mi2)};
  fileName = testDataFolder + "\\" + "lstm_0_W_quantized" + ".const";
  std::vector<WgT> lstm0_w_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (lstm0_w_weight.size() !=
      (lstm0_w_weightShape[0] * lstm0_w_weightShape[1] *
       lstm0_w_weightShape[2])) {
    std::cout
        << "lstm_0_W weight parameter file is not proper. Expected size = "
        << (lstm0_w_weightShape[0] * lstm0_w_weightShape[1] *
            lstm0_w_weightShape[2])
        << ", Actual Size = " << lstm0_w_weight.size() << std::endl;
  }
  std::vector<size_t> lstm1_w_weightShape = {2, 512, static_cast<size_t>(Mo2)};
  fileName = testDataFolder + "\\" + "lstm_1_W_quantized" + ".const";
  std::vector<WgT> lstm1_w_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (lstm1_w_weight.size() !=
      (lstm1_w_weightShape[0] * lstm1_w_weightShape[1] *
       lstm1_w_weightShape[2])) {
    std::cout
        << "lstm_1_W weight parameter file is not proper. Expected size = "
        << (lstm1_w_weightShape[0] * lstm1_w_weightShape[1] *
            lstm1_w_weightShape[2])
        << ", Actual Size = " << lstm1_w_weight.size() << std::endl;
  }
  std::vector<size_t> r_weightShape = {2, 512, 128};
  fileName = testDataFolder + "\\" + "lstm_0_R_quantized" + ".const";
  std::vector<WgT> lstm0_r_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (lstm0_r_weight.size() !=
      (r_weightShape[0] * r_weightShape[1] * r_weightShape[2])) {
    std::cout
        << "lstm_0_R weight parameter file is not proper. Expected size = "
        << (r_weightShape[0] * r_weightShape[1] * r_weightShape[2])
        << ", Actual Size = " << lstm0_r_weight.size() << std::endl;
  }

  fileName = testDataFolder + "\\" + "lstm_1_R_quantized" + ".const";
  std::vector<WgT> lstm1_r_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (lstm1_r_weight.size() !=
      (r_weightShape[0] * r_weightShape[1] * r_weightShape[2])) {
    std::cout
        << "lstm_1_R weight parameter file is not proper. Expected size = "
        << (r_weightShape[0] * r_weightShape[1] * r_weightShape[2])
        << ", Actual Size = " << lstm1_r_weight.size() << std::endl;
  }
  std::vector<size_t> b_weightShape = {2, 1024};
  fileName = testDataFolder + "\\" + "lstm_0_B_quantized" + ".const";
  std::vector<WgT> lstm0_b_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (lstm0_b_weight.size() != (b_weightShape[0] * b_weightShape[1])) {
    std::cout
        << "lstm_0_B weight parameter file is not proper. Expected size = "
        << (b_weightShape[0] * b_weightShape[1])
        << ", Actual Size = " << lstm0_b_weight.size() << std::endl;
  }
  fileName = testDataFolder + "\\" + "lstm_1_B_quantized" + ".const";
  std::vector<WgT> lstm1_b_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (lstm1_b_weight.size() != (b_weightShape[0] * b_weightShape[1])) {
    std::cout
        << "lstm_1_B weight parameter file is not proper. Expected size = "
        << (b_weightShape[0] * b_weightShape[1])
        << ", Actual Size = " << lstm1_b_weight.size() << std::endl;
  }

  std::vector<size_t> ifmShape = {static_cast<size_t>(Mi0),
                                  static_cast<size_t>(Mi1),
                                  static_cast<size_t>(Mi2)};
  auto ifmSize = ifmShape[0] * ifmShape[1] * ifmShape[2];
  fileName = testDataFolder + "\\" + "ifm" + ".bin";
  std::vector<InT> ifm = OpsFusion::read_bin_file<InT>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }

  std::vector<size_t> ofmShape = {static_cast<size_t>(Mo0),
                                  static_cast<size_t>(Mo1),
                                  static_cast<size_t>(Mo2)};
  int32_t garbage_value = 0xAAAABBBB;
  std::vector<OuT> ofm(Mo0 * Mo1 * Mo2, garbage_value);

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{Mi0, Mi1, Mi2};
  attr["scales"] = scales_zps[std::to_string(modelNum)].scales;
  attr["zero_points"] = scales_zps[std::to_string(modelNum)].zps;

  ryzenai::lstm lstm_ = ryzenai::lstm<InT, WgT, OuT>(ifmDtype, weightDtype,
                                                     ofmDtype, false, attr);
  debug = true;
  lstm_.debug(debug);
  std::vector<size_t> weightShape = {1, 1, 672512};
  lstm_.set_params(modelNum, ifmShape, weightShape, ofmShape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{lstm0_w_weight.data(), lstm0_w_weightShape, weightDtype},
                  {lstm0_r_weight.data(), r_weightShape, weightDtype},
                  {lstm0_b_weight.data(), b_weightShape, weightDtype},
                  {lstm1_w_weight.data(), lstm1_w_weightShape, weightDtype},
                  {lstm1_r_weight.data(), r_weightShape, weightDtype},
                  {lstm1_b_weight.data(), b_weightShape, weightDtype}};
  lstm_.initialize_const_params(const_Tensor, attr);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{ifm.data(), ifmShape, ifmDtype}};
  std::vector<Tensor> output_Tensor = {{ofm.data(), ofmShape, ofmDtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("Mi0=" << Mi0 << ", Mi1=" << Mi1 << ", Mo0=" << Mo0
                  << ", Mo1=" << Mo0);
  PROFILE_THIS(lstm_.execute(input_Tensor, output_Tensor));
#else
  lstm_.execute(input_Tensor, output_Tensor);
#endif

  generatedFileName = testDataFolder + "\\" + "ofmOut" + ".txt";
  write32BitHexTxtFile(generatedFileName, (OuT *)ofm.data(), ofm.size());

  fileName = testDataFolder + "\\" + "ofm32_ref.txt";
  if (CompareFileContents(fileName, generatedFileName)) {
    std::cout << "Error: ofm output doesn't match" << std::endl;
    err_count++;
  }
  return err_count;
}

/* mswbjvw-320 */
TEST(LstmTesta16w16c16, Kernel1) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      80, 1, 64, 80, 2, 256, false, "uint16", "uint16", "uint16", 320);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* mswbjvw-640 */
TEST(LstmTesta16w16c16, Kernel2) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      160, 1, 64, 160, 2, 256, false, "uint16", "uint16", "uint16", 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* mswbjvw-1280 */
TEST(LstmTesta16w16c16, Kernel3) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      320, 1, 64, 320, 2, 256, false, "uint16", "uint16", "uint16", 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
