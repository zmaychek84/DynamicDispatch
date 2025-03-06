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

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/conv/conv.hpp>
#include <ops/ops_common/help_file.hpp>
#include <ops/ops_common/matmul_matrix.hpp>

#include "enable_perf.hpp"
#include "test_common.hpp"

using namespace std;

static std::string GetTestSubFolderName(std::string prefix, int zeroPoint,
                                        int K, int N, int F0) {
  return prefix + "_" + std::to_string(zeroPoint) + "_" + std::to_string(F0) +
         "_" + std::to_string(K) + "_" + std::to_string(N);
}

static std::string GetParamKey(std::string prefix, int64_t zp, int64_t K,
                               int64_t N, int64_t F0) {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(F0) + "_" +
         std::to_string(K) + "_" + std::to_string(N);
}

static int CompareFileContents(const std::string &input_file_path,
                               const std::string &output_file_path) {
  // Open the input file and read the contents into a string
  std::ifstream input_file(input_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_file_path << std::endl;
    return -1;
  }

  std::string input_file_contents((std::istreambuf_iterator<char>(input_file)),
                                  std::istreambuf_iterator<char>());
  input_file.close();

  // Open the output file and read the contents into a string
  std::ifstream output_file(output_file_path);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_file_path
              << std::endl;
    return -1;
  }

  std::string output_file_contents(
      (std::istreambuf_iterator<char>(output_file)),
      std::istreambuf_iterator<char>());
  output_file.close();

  // Compare the two file contents
  if (input_file_contents == output_file_contents) {
    return 0;
  } else {
    return -1;
  }
}

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>

/**
 *
 * @tparam InT
 * @tparam WgT
 * @tparam OuT
 * @param Mi0         Model Input dim 0
 * @param Mi1         Model Input dim 1
 * @param F0          Filter dim 0
 * @param F1          Filter dim 1
 * @param K           No. of input channels
 * @param N           No. of output channels
 * @param Mo0         Model Output 0
 * @param Mo1         Model Output 1
 * @param groupId
 * @param debug
 * @param ifmDtype
 * @param weightDtype
 * @param ofmDtype
 * @param zeroPoint
 * @param modelName
 * @return
 */
int test_conv(int Mi0, int Mi1, int F0, int F1, int K, int N, int Mo0, int Mo1,
              int groupId, bool debug = false,
              const string &ifmDtype = "uint16",
              const string &weightDtype = "uint8",
              const string &ofmDtype = "uint16", int zeroPoint = 1,
              const string &modelName = "m3uec", bool useTxnBinWithZp = true,
              int width = 0) {
  int err_count = 0;
  std::string fileName, testDataFolder, generatedFileName;
  std::string modelNameLowerCase = modelName;

  std::transform(modelNameLowerCase.begin(), modelNameLowerCase.end(),
                 modelNameLowerCase.begin(), ::tolower);
  testDataFolder =
      OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(modelNameLowerCase, zeroPoint, K, N, F0);
  std::vector<size_t> weightShape;

  if (groupId == 1) {
    weightShape = {static_cast<size_t>(N), static_cast<size_t>(K),
                   static_cast<size_t>(F0), static_cast<size_t>(F1)}; // weight
  } else {
    weightShape = {static_cast<size_t>(N), static_cast<size_t>(1),
                   static_cast<size_t>(F0), static_cast<size_t>(F1)}; // weight
  }
  fileName = testDataFolder + "\\" + "weight" + ".const";
  std::vector<WgT> weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (weight.size() !=
      (weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (weightShape[0] * weightShape[1] * weightShape[2] *
                  weightShape[3])
              << ", Actual Size = " << weight.size() << std::endl;
  }

  std::vector<size_t> ifmShape;
  size_t ifmSize;
  if ((zeroPoint == 29172) && (modelNameLowerCase == "m3uec")) {
    /* This is a specific case required for layer 1 of m3uec model only */
    ifmShape = {1, (static_cast<size_t>(Mi0) * static_cast<size_t>(Mi1)),
                static_cast<size_t>(K + 1)}; // activate
    ifmSize = (K + 1) * Mi0 * Mi1;
  } else if ((zeroPoint == 40597) && ((modelNameLowerCase == "mswbjvw640") ||
                                      (modelNameLowerCase == "mswbjvw1280") ||
                                      (modelNameLowerCase == "mswbjvw2560"))) {
    /* This is a specific case required for layer 1 of PSO640 model only */
    ifmShape = {1, (static_cast<size_t>(Mi0) * static_cast<size_t>(Mi1)),
                static_cast<size_t>(K - 4)}; // activate
    ifmSize = (K - 4) * Mi0 * Mi1;
  } else {
    ifmShape = {1, (static_cast<size_t>(Mi0) * static_cast<size_t>(Mi1)),
                static_cast<size_t>(K)}; // activate
    ifmSize = K * Mi0 * Mi1;
  }
  fileName = testDataFolder + "\\" + "ifm" + ".const";
  std::vector<InT> ifm = OpsFusion::read_bin_file<InT>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }

  std::vector<size_t> ofmShape = {
      1, (static_cast<size_t>(Mo0) * static_cast<size_t>(Mo1)),
      static_cast<size_t>(N)};
  int32_t garbage_value = 0xAAAABBBB;
  std::vector<OuT> ofm(N * (Mo0) * (Mo1), garbage_value);

  std::map<std::string, std::any> attr;
  attr["group"] = std::vector<int>{groupId};
  attr["input_shape"] = std::vector<int>{1, K, Mi0, Mi1};
  attr["output_shape"] = std::vector<int>{1, N, Mo0, Mo1};
  int weightThirdDim;
  if (groupId == 1) {
    weightThirdDim = K;
  } else {
    weightThirdDim = 1;
  }

  /* second dimension in weight shape we are making there variable.
  in actual use case it will come from onnx. Which is K for conv kernel and 1
  for DWC*/
  attr["weight_shape"] = std::vector<int>{N, weightThirdDim, F0, F1};
  attr["zero_point"] = std::vector<int>{zeroPoint};
  if (width != 0) {
    attr["width"] = std::vector<int>{width};
  }

  ryzenai::conv conv_ = ryzenai::conv<InT, WgT, OuT>(ifmDtype, weightDtype,
                                                     ofmDtype, false, attr);
  debug = true;
  conv_.debug(debug);
  conv_.set_params(modelNameLowerCase, useTxnBinWithZp);

  /*
   * qdq contains c0
   * qdq_params contains c1, c2 and shift_out
   */
  srand(0xABCD);
  std::vector<int64_t> qdq(1 * N); // c0
  initialize_random<int64_t>(qdq, 1 * N, 32, 0);

  // TODO: verify these parameters
  uint32_t C1 = 0;
  uint32_t C2 = 10;
  uint32_t shift_out = 0;

  std::vector<size_t> qdq_shape = {static_cast<unsigned long long>(N)};
  std::vector<size_t> qdq_params_shape = {3}; // c1, c2, shift_out
  std::vector<int32_t> qdq_params(matmul_matrix::QDQparam_size);
  qdq_params[0] = C1;
  qdq_params[1] = C2;
  qdq_params[2] = shift_out;

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{weight.data(), weightShape, weightDtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};
  conv_.initialize_const_params(const_Tensor, attr);

  if (debug == true) {
    fileName = testDataFolder + "\\" + "wtsRef.txt";
    std::string weightGeneratedFolder =
        OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
        "unit_tests" + "\\" + "testDataMladf" + "\\" + "GeneratedWeights";

    if (zeroPoint == 29172) {
      generatedFileName =
          weightGeneratedFolder + "\\" +
          GetParamKey("wtsGenerated", zeroPoint, (K + 1), N, F0) + ".txt";
    } else {
      generatedFileName = weightGeneratedFolder + "\\" +
                          GetParamKey("wtsGenerated", zeroPoint, K, N, F0) +
                          ".txt";
    }
    if (CompareFileContents(fileName, generatedFileName)) {
      std::cout << "Error: the weight generated are not proper" << std::endl;
      err_count++;
    }
  }

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{ifm.data(), ifmShape, ifmDtype}};
  std::vector<Tensor> output_Tensor = {{ofm.data(), ofmShape, ofmDtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("Mi0=" << Mi0 << ", Mi1=" << Mi1 << ", F0=" << F0 << ", F1=" << F1
                  << ", K=" << K << ", N=" << N << ", Mo0=" << Mo0
                  << ", Mo1=" << Mo0);
  PROFILE_THIS(conv_.execute(input_Tensor, output_Tensor));
#else
  conv_.execute(input_Tensor, output_Tensor);
#endif

  generatedFileName = testDataFolder + "\\" + "ofmOut" + ".txt";
  write32BitHexTxtFile(generatedFileName, (OuT *)ofm.data(), ofm.size());

  fileName = testDataFolder + "\\" + "ofmRef.txt";
  if (CompareFileContents(fileName, generatedFileName)) {
    std::cout << "Error: ofm output doesn't match" << std::endl;
    err_count++;
  }
  return err_count;
}

#if 0
/* PST  under development */

/* PSS Under Development */
TEST(ConvTesta16w8c16_, PssKernel29) {
  /* decoder.up_blocks.3.resnets.0.conv_shortcut */
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      512, 512, 1, 1, 256, 128, 512, 512, 1, false, "uint16", "uint8", "uint16",
      37529, "PSS", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif

#if 0
/* PST Verified */
TEST(ConvTesta16w8c16_, PstKernel20) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      64, 64, 3, 3, 512, 512, 64, 64, 1, false, "uint16", "uint8", "uint16",
      699, "PST", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(ConvTesta16w8c16_, PstKernel28) {
  /* quant_conv */
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      64, 64, 1, 1, 8, 8, 64, 64, 1, false, "uint16", "uint8", "uint16", 29706,
      "PST", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, PstKernel1) {
  /* encoder.down_blocks.2.resnets.0.conv_shortcut */
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      128, 128, 1, 1, 256, 512, 128, 128, 1, false, "uint16", "uint8", "uint16",
      19793, "PST", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, PstKernel7) {
  /* encoder.down_blocks.1.resnets.0.conv_shortcut */
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      256, 256, 1, 1, 128, 256, 256, 256, 1, false, "uint16", "uint8", "uint16",
      20675, "PST", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* PSS Verified */
TEST(ConvTesta16w8c16_, PssKernel21) {
  /* decoder.up_blocks.2.resnets.0.conv_shortcut */
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      256, 256, 1, 1, 512, 256, 256, 256, 1, false, "uint16", "uint8", "uint16",
      37147, "PSS", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

#endif
/* mswbjvw 320 */
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso2320Kernel1) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      60, 320, 3, 3, 8, 16, 32, 160, 1, false, "uint16", "uint16", "uint16",
      40597, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

#if 0
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso2320Kernel2) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      32, 160, 3, 3, 16, 32, 16, 80, 1, false, "uint16", "uint16", "uint16",
      32705, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel3) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 80, 1, 1, 32, 16, 16, 80, 1, false, "uint16", "uint16", "uint16",
      36423, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel4) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16", "uint16", "uint16",
      33409, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel5) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 80, 1, 1, 32, 128, 16, 80, 1, false, "uint16", "uint16", "uint16",
      29586, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel6) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 80, 1, 1, 128, 16, 16, 80, 1, false, "uint16", "uint16", "uint16",
      25513, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel7) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16", "uint16", "uint16",
      31530, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso2320Kernel8) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 80, 1, 1, 32, 128, 8, 80, 1, false, "uint16", "uint16", "uint16",
      32591, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel9) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 80, 1, 1, 128, 32, 8, 80, 1, false, "uint16", "uint16", "uint16",
      31990, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel10) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16", "uint16", "uint16", 36064,
      "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel11) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 80, 1, 1, 48, 256, 8, 80, 1, false, "uint16", "uint16", "uint16",
      35326, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel12) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 80, 1, 1, 256, 32, 8, 80, 1, false, "uint16", "uint16", "uint16",
      34702, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel13) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16", "uint16", "uint16", 30051,
      "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso2320Kernel14) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 80, 1, 1, 48, 256, 4, 80, 1, false, "uint16", "uint16", "uint16",
      35719, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel15) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 1, 1, 256, 64, 4, 80, 1, false, "uint16", "uint16", "uint16",
      26536, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel16) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16", "uint16", "uint16", 22444,
      "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel17) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16", "uint16", "uint16",
      32234, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel18) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 1, 1, 512, 64, 4, 80, 1, false, "uint16", "uint16", "uint16",
      33891, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel19) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16", "uint16", "uint16", 33497,
      "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel20) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16", "uint16", "uint16",
      31960, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2320Kernel21) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 80, 1, 1, 512, 16, 4, 80, 1, false, "uint16", "uint16", "uint16",
      33774, "mswbjvw");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* mswbjvw 640 */
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso2640Kernel1) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      60, 640, 3, 3, 8, 16, 32, 320, 1, false, "uint16", "uint16", "uint16",
      40597, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso2640Kernel2) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      32, 320, 3, 3, 16, 32, 16, 160, 1, false, "uint16", "uint16", "uint16",
      32705, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel3) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 160, 1, 1, 32, 16, 16, 160, 1, false, "uint16", "uint16", "uint16",
      36423, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel4) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 160, 3, 3, 16, 32, 16, 160, 1, false, "uint16", "uint16", "uint16",
      33409, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel5) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 160, 1, 1, 32, 128, 16, 160, 1, false, "uint16", "uint16", "uint16",
      29586, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel6) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 160, 1, 1, 128, 16, 16, 160, 1, false, "uint16", "uint16", "uint16",
      25513, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel7) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 160, 3, 3, 16, 32, 16, 160, 1, false, "uint16", "uint16", "uint16",
      31530, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso2640Kernel8) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 160, 1, 1, 32, 128, 8, 160, 1, false, "uint16", "uint16", "uint16",
      32591, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel9) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 160, 1, 1, 128, 32, 8, 160, 1, false, "uint16", "uint16", "uint16",
      31990, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel10) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 160, 3, 3, 32, 48, 8, 160, 1, false, "uint16", "uint16", "uint16",
      36064, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel11) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 160, 1, 1, 48, 256, 8, 160, 1, false, "uint16", "uint16", "uint16",
      35326, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel12) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 160, 1, 1, 256, 32, 8, 160, 1, false, "uint16", "uint16", "uint16",
      34702, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel13) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 160, 3, 3, 32, 48, 8, 160, 1, false, "uint16", "uint16", "uint16",
      30051, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso2640Kernel14) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 160, 1, 1, 48, 256, 4, 160, 1, false, "uint16", "uint16", "uint16",
      35719, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel15) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 1, 1, 256, 64, 4, 160, 1, false, "uint16", "uint16", "uint16",
      26536, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel16) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 3, 3, 64, 80, 4, 160, 1, false, "uint16", "uint16", "uint16",
      22444, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel17) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 1, 1, 80, 512, 4, 160, 1, false, "uint16", "uint16", "uint16",
      32234, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel18) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 1, 1, 512, 64, 4, 160, 1, false, "uint16", "uint16", "uint16",
      33891, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel19) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 3, 3, 64, 80, 4, 160, 1, false, "uint16", "uint16", "uint16",
      33497, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel20) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 1, 1, 80, 512, 4, 160, 1, false, "uint16", "uint16", "uint16",
      31960, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso2640Kernel21) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 160, 1, 1, 512, 16, 4, 160, 1, false, "uint16", "uint16", "uint16",
      33774, "mswbjvw640", true, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* mswbjvw 1280 */
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso21280Kernel1) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      60, 1280, 3, 3, 8, 16, 32, 640, 1, false, "uint16", "uint16", "uint16",
      40597, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso21280Kernel2) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      32, 640, 3, 3, 16, 32, 16, 320, 1, false, "uint16", "uint16", "uint16",
      32705, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel3) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 320, 1, 1, 32, 16, 16, 320, 1, false, "uint16", "uint16", "uint16",
      36423, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel4) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 320, 3, 3, 16, 32, 16, 320, 1, false, "uint16", "uint16", "uint16",
      33409, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel5) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 320, 1, 1, 32, 128, 16, 320, 1, false, "uint16", "uint16", "uint16",
      29586, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel6) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 320, 1, 1, 128, 16, 16, 320, 1, false, "uint16", "uint16", "uint16",
      25513, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel7) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 320, 3, 3, 16, 32, 16, 320, 1, false, "uint16", "uint16", "uint16",
      31530, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso21280Kernel8) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 320, 1, 1, 32, 128, 8, 320, 1, false, "uint16", "uint16", "uint16",
      32591, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel9) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 320, 1, 1, 128, 32, 8, 320, 1, false, "uint16", "uint16", "uint16",
      31990, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel10) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 320, 3, 3, 32, 48, 8, 320, 1, false, "uint16", "uint16", "uint16",
      36064, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel11) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 320, 1, 1, 48, 256, 8, 320, 1, false, "uint16", "uint16", "uint16",
      35326, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel12) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 320, 1, 1, 256, 32, 8, 320, 1, false, "uint16", "uint16", "uint16",
      34702, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel13) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 320, 3, 3, 32, 48, 8, 320, 1, false, "uint16", "uint16", "uint16",
      30051, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso21280Kernel14) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 320, 1, 1, 48, 256, 4, 320, 1, false, "uint16", "uint16", "uint16",
      35719, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel15) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 1, 1, 256, 64, 4, 320, 1, false, "uint16", "uint16", "uint16",
      26536, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel16) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 3, 3, 64, 80, 4, 320, 1, false, "uint16", "uint16", "uint16",
      22444, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel17) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 1, 1, 80, 512, 4, 320, 1, false, "uint16", "uint16", "uint16",
      32234, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel18) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 1, 1, 512, 64, 4, 320, 1, false, "uint16", "uint16", "uint16",
      33891, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel19) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 3, 3, 64, 80, 4, 320, 1, false, "uint16", "uint16", "uint16",
      33497, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel20) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 1, 1, 80, 512, 4, 320, 1, false, "uint16", "uint16", "uint16",
      31960, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso21280Kernel21) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 320, 1, 1, 512, 16, 4, 320, 1, false, "uint16", "uint16", "uint16",
      33774, "mswbjvw1280", true, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* mswbjvw 1280 */
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso22560Kernel1) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      60, 2560, 3, 3, 8, 16, 32, 1280, 1, false, "uint16", "uint16", "uint16",
      40597, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 2]*/
TEST(ConvTesta16w16c16_, Pso22560Kernel2) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      32, 1280, 3, 3, 16, 32, 16, 640, 1, false, "uint16", "uint16", "uint16",
      32705, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel3) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 640, 1, 1, 32, 16, 16, 640, 1, false, "uint16", "uint16", "uint16",
      36423, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel4) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 640, 3, 3, 16, 32, 16, 640, 1, false, "uint16", "uint16", "uint16",
      33409, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel5) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 640, 1, 1, 32, 128, 16, 640, 1, false, "uint16", "uint16", "uint16",
      29586, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel6) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 640, 1, 1, 128, 16, 16, 640, 1, false, "uint16", "uint16", "uint16",
      25513, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel7) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 640, 3, 3, 16, 32, 16, 640, 1, false, "uint16", "uint16", "uint16",
      31530, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso22560Kernel8) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      16, 640, 1, 1, 32, 128, 8, 640, 1, false, "uint16", "uint16", "uint16",
      32591, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel9) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 640, 1, 1, 128, 32, 8, 640, 1, false, "uint16", "uint16", "uint16",
      31990, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel10) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 640, 3, 3, 32, 48, 8, 640, 1, false, "uint16", "uint16", "uint16",
      36064, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel11) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 640, 1, 1, 48, 256, 8, 640, 1, false, "uint16", "uint16", "uint16",
      35326, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel12) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 640, 1, 1, 256, 32, 8, 640, 1, false, "uint16", "uint16", "uint16",
      34702, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel13) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 640, 3, 3, 32, 48, 8, 640, 1, false, "uint16", "uint16", "uint16",
      30051, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* conv + maxpool fusion stride is [2, 1]*/
TEST(ConvTesta16w16c16_, Pso22560Kernel14) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      8, 640, 1, 1, 48, 256, 4, 640, 1, false, "uint16", "uint16", "uint16",
      35719, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel15) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 1, 1, 256, 64, 4, 640, 1, false, "uint16", "uint16", "uint16",
      26536, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel16) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 3, 3, 64, 80, 4, 640, 1, false, "uint16", "uint16", "uint16",
      22444, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel17) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 1, 1, 80, 512, 4, 640, 1, false, "uint16", "uint16", "uint16",
      32234, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel18) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 1, 1, 512, 64, 4, 640, 1, false, "uint16", "uint16", "uint16",
      33891, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel19) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 3, 3, 64, 80, 4, 640, 1, false, "uint16", "uint16", "uint16",
      33497, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel20) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 1, 1, 80, 512, 4, 640, 1, false, "uint16", "uint16", "uint16",
      31960, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w16c16_, Pso22560Kernel21) {
  int err_count = test_conv<uint16_t, uint16_t, uint16_t>(
      4, 640, 1, 1, 512, 16, 4, 640, 1, false, "uint16", "uint16", "uint16",
      33774, "mswbjvw2560", true, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Below are m3uec Kernels working with ConvDwcGap_Psi.xclbin */
TEST(ConvTesta16w8c16_, Kernel1) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      224, 224, 7, 7, 3, 128, 56, 56, 1, false, "uint16", "uint8", "uint16",
      29172, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel48) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 1024, 7, 7, 1, false, "uint16", "uint8", "uint16",
      37978, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel30) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      10008, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(ConvTesta16w8c16_, Kernel17) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      10240, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel28) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      13671, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel20) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      16932, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel24) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      22359, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel27) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      22529, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel26) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      22886, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel22) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      23805, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel29) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      2469, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel51) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      7, 7, 3, 3, 1024, 1024, 7, 7, 1024, false, "uint16", "uint8", "uint16",
      24764, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel16) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      25504, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel41) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      2641, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel45) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      2643, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel47) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      26744, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel14) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      26808, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel33) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      2774, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel21) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      2777, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel19) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      28057, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel12) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      28119, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel50) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      7, 7, 3, 3, 1024, 1024, 7, 7, 1024, false, "uint16", "uint8", "uint16",
      28167, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel15) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      28348, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel35) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      28707, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel31) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      28928, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel39) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      28982, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel37) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      2909, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel43) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      29442, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel3) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      56, 56, 3, 3, 128, 128, 56, 56, 128, false, "uint16", "uint8", "uint16",
      31078, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel18) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      31118, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel8) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      28, 28, 3, 3, 256, 256, 28, 28, 256, false, "uint16", "uint8", "uint16",
      31330, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel11) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      28, 28, 3, 3, 256, 512, 14, 14, 1, false, "uint16", "uint8", "uint16",
      32479, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel6) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      56, 56, 3, 3, 128, 256, 28, 28, 1, false, "uint16", "uint8", "uint16",
      32895, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel2) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      56, 56, 3, 3, 128, 128, 56, 56, 128, false, "uint16", "uint8", "uint16",
      33438, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel9) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      28, 28, 3, 3, 256, 256, 28, 28, 256, false, "uint16", "uint8", "uint16",
      33977, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel4) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      56, 56, 3, 3, 128, 128, 56, 56, 128, false, "uint16", "uint8", "uint16",
      34324, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel5) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      56, 56, 3, 3, 128, 128, 56, 56, 128, false, "uint16", "uint8", "uint16",
      37709, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel49) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      7, 7, 3, 3, 1024, 1024, 7, 7, 1024, false, "uint16", "uint8", "uint16",
      39361, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel25) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      4004, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel52) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      7, 7, 3, 3, 1024, 1024, 7, 7, 1024, false, "uint16", "uint8", "uint16",
      40321, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel7) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      28, 28, 3, 3, 256, 256, 28, 28, 256, false, "uint16", "uint8", "uint16",
      40592, "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel34) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      4117, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel38) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      4653, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel10) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      28, 28, 3, 3, 256, 256, 28, 28, 256, false, "uint16", "uint8", "uint16",
      46664, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel42) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      4767, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel40) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      5316, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel44) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      5888, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel36) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      6459, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel46) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      7414, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel13) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      7631, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel32) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      8194, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(ConvTesta16w8c16_, Kernel23) {
  int err_count = test_conv<uint16_t, uint8_t, uint16_t>(
      14, 14, 3, 3, 512, 512, 14, 14, 512, false, "uint16", "uint8", "uint16",
      8568, "m3uec", false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif
