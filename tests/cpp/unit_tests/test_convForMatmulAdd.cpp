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
#include <ops/convForMatmulAdd/convForMatmulAdd.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

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

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_convForMatumulAdd(int Mi0, int Mi1, int F0, int F1, int K, int N,
                           int Mo0, int Mo1, int groupId, int wts_zp, int c1,
                           int c2, bool debug = false,
                           const std::string &ifmDtype = "uint16",
                           const std::string &weightDtype = "uint16",
                           const std::string &ofmDtype = "uint16",
                           int zeroPoint = 1,
                           const std::string &modelName = "m3uec",
                           bool useTxnBinWithZp = true, int width = 0,
                           int graph_id = 0) {
  int err_count = 0;
  std::string fileName, testDataFolder, generatedFileName;
  std::string modelNameLowerCase = modelName;

  std::transform(modelNameLowerCase.begin(), modelNameLowerCase.end(),
                 modelNameLowerCase.begin(), ::tolower);
  testDataFolder =
      OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(modelNameLowerCase, zeroPoint, K, N, F0);
  std::cout << testDataFolder << "\n";
  std::vector<size_t> weightShape;

  weightShape = {static_cast<size_t>(K), static_cast<size_t>(N)}; // weight
  fileName = testDataFolder + "\\" + "weight" + ".const";
  std::vector<WgT> weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (weight.size() != (weightShape[0] * weightShape[1])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (weightShape[0] * weightShape[1])
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
  attr["input_shape"] = std::vector<int>{Mi1, Mi0, K};
  attr["output_shape"] = std::vector<int>{Mo1, Mo0, N};
  attr["weight_shape"] = std::vector<int>{F0, F1};
  attr["zero_point"] = wts_zp;
  attr["c1"] = c1;
  attr["c2"] = c2;

  int weightThirdDim;
  if (groupId == 1) {
    weightThirdDim = K;
  } else {
    weightThirdDim = 1;
  }
  /* second dimention in weight shape we are making there variable.
  in actual use case it will come from onnx. Which is K for convForMatmulAdd
  kernel and 1 for DWC*/
  attr["weight_shape"] = std::vector<int>{weightThirdDim, N};
  // attr["zero_point"] = std::vector<int>{zeroPoint};
  attr["graph_id"] = graph_id;
  if (width != 0) {
    attr["width"] = std::vector<int>{width};
  }
  ryzenai::convForMatmulAdd convForMatmulAdd_ =
      ryzenai::convForMatmulAdd<InT, WgT, OuT>(ifmDtype, weightDtype, ofmDtype,
                                               false, attr);
  debug = true;
  convForMatmulAdd_.debug(debug);
  convForMatmulAdd_.set_params(modelNameLowerCase, useTxnBinWithZp);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{weight.data(), weightShape, weightDtype}};
  convForMatmulAdd_.initialize_const_params(const_Tensor, attr);

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
#if 0
  /* Currently we don't have txn bin file for stand alone fc layer teseting. That is why below execution is commented */
#ifdef UNIT_TEST_PERF
  LOG_THIS("Mi0=" << Mi0 << ", Mi1=" << Mi1 << ", F0=" << F0 << ", F1=" << F1
                  << ", K=" << K << ", N=" << N << ", Mo0=" << Mo0
                  << ", Mo1=" << Mo0);
  PROFILE_THIS(convForMatmulAdd_.execute(input_Tensor, output_Tensor));
#else
  convForMatmulAdd_.execute(input_Tensor, output_Tensor);
#endif

  generatedFileName = testDataFolder + "\\" + "ofmOut" + ".txt";
  write32BitHexTxtFile(generatedFileName, (OuT *)ofm.data(), ofm.size());

  fileName = testDataFolder + "\\" + "ofmRef.txt";
  if (CompareFileContents(fileName, generatedFileName)) {
    std::cout << "Error: ofm output doesn't match" << std::endl;
    err_count++;
  }
#endif
  return err_count;
}
#if 0
TEST(ConvTesta16w8c16_, Pso2_2560LayerFc) {
  int err_count = test_convForMatumulAdd<uint16_t, uint16_t, uint16_t>(
      1, 640, 1, 1, 256, 548, 1, 640, 1, 27512, -1003328, 4668, false, "uint16",
      "uint16", "uint16", 32756, "mswbjvw2560", true, 2560, 2560);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(ConvTesta16w8c16_, Pso2_1280LayerFc) {
  int err_count = test_convForMatumulAdd<uint16_t, uint16_t, uint16_t>(
      1, 320, 1, 1, 256, 548, 1, 320, 1, 27512, -1018804, 4740, false, "uint16",
      "uint16", "uint16", 32760, "mswbjvw1280", true, 1280, 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(ConvTesta16w8c16_, Pso2_640LayerFc) {
  int err_count = test_convForMatumulAdd<uint16_t, uint16_t, uint16_t>(
      1, 160, 1, 1, 256, 548, 1, 160, 1, 27512, -952388, 4431, false, "uint16",
      "uint16", "uint16", 32761, "mswbjvw640", true, 640, 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(ConvTesta16w8c16_, Pso2_320LayerFc) {
  int err_count = test_convForMatumulAdd<uint16_t, uint16_t, uint16_t>(
      1, 80, 1, 1, 256, 548, 1, 80, 1, 27512, -990217, 4607, false, "uint16",
      "uint16", "uint16", 32771, "mswbjvw", true, 0, 320);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif
