/*
 * Copyright � 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/concateOps/concateOps.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

using namespace std;

/* ParamsStruct to store extracted parameters */
struct ParamsStruct {
  int Mi0, Mi1, F0, F1, K, N, Mo0, Mo1, groupId, width;
  bool debug, useTxnBinWithZp;
  std::string opIfmDtype, opWtsDtype, opOfmDtype;
  int zeroPoint;
  std::string opModelName, operatorName;
};

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
  /* Open the input file and read the contents into a string */
  std::ifstream input_file(input_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_file_path << std::endl;
    return -1;
  }

  /* Open the output file and read the contents into a string */
  std::ifstream output_file(output_file_path);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_file_path
              << std::endl;
    return -1;
  }

  /* Compare the two file contents line by line */
  std::string input_line, output_line;
  std::size_t line_number = 1;
  bool files_are_different = false;

  while (std::getline(input_file, input_line) &&
         std::getline(output_file, output_line)) {
    if (input_line != output_line) {
      std::cout << "Mismatch at line " << line_number << ":" << std::endl;
      std::cout << "  Input file:  " << input_line << std::endl;
      std::cout << "  Output file: " << output_line << std::endl;
      files_are_different = true;
    }
    ++line_number;
  }

  if (input_file.bad() || output_file.bad()) {
    std::cerr << "Error while reading the files." << std::endl;
    return -1;
  }

  if (std::getline(input_file, input_line) ||
      std::getline(output_file, output_line)) {
    std::cerr << "Files have different number of lines." << std::endl;
    return -1;
  }

  input_file.close();
  output_file.close();

  if (files_are_different) {
    return -1;
  } else {
    return 0;
  }
}

/* Helper function to convert a string to lowercase */
static std::string toLowercase(const std::string &input) {
  std::string lowercase = input;
  std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
                 ::tolower);
  return lowercase;
}

/* Helper function to extract parameters */
static ParamsStruct ExtractParameters(
    const std::tuple<int, int, int, int, int, int, int, int, int, bool,
                     std::string, std::string, std::string, int, std::string,
                     bool, int, std::string> &params) {
  ParamsStruct ps;
  ps.Mi0 = std::get<0>(params);
  ps.Mi1 = std::get<1>(params);
  ps.F0 = std::get<2>(params);
  ps.F1 = std::get<3>(params);
  ps.K = std::get<4>(params);
  ps.N = std::get<5>(params);
  ps.Mo0 = std::get<6>(params);
  ps.Mo1 = std::get<7>(params);
  ps.groupId = std::get<8>(params);
  ps.debug = std::get<9>(params);
  ps.opIfmDtype = toLowercase(std::get<10>(params));
  ps.opWtsDtype = toLowercase(std::get<11>(params));
  ps.opOfmDtype = toLowercase(std::get<12>(params));
  ps.zeroPoint = std::get<13>(params);
  ps.opModelName = toLowercase(std::get<14>(params));
  ps.useTxnBinWithZp = std::get<15>(params);
  ps.width = std::get<16>(params);
  ps.operatorName = toLowercase(std::get<17>(params));
  return ps;
}

/* Helper function to initialize ofm data buffer with garbase values */
static Tensor
GetOfmTensor(const ParamsStruct &ps,
             std::vector<std::vector<uint16_t>> &ofmDataContainer) {
  std::vector<size_t> ofmShape = {
      1, (static_cast<size_t>(ps.Mo0) * static_cast<size_t>(ps.Mo1)),
      static_cast<size_t>(ps.N)};
  int32_t garbage_value = 0xAAAABBBB;
  /* Sandip TBD : Need to replace vector type from opOfmDtype */
  std::vector<uint16_t> ofm(ps.N * (ps.Mo0) * (ps.Mo1), garbage_value);

  /* Add the ofm data to the ofmDataContainer */
  ofmDataContainer.push_back(std::move(ofm));

  /* Get a reference to the last element of the weightDataContainer which
  contains the weight data just added */
  std::vector<uint16_t> &ofm_data_ref = ofmDataContainer.back();
  return {ofm_data_ref.data(), ofmShape, ps.opOfmDtype};
}

/* Helper function to read ifm data from a file */
static Tensor
GetIfmTensor(const ParamsStruct &ps,
             std::vector<std::vector<uint16_t>> &ifmDataContainer) {
  std::string testDataFolder =
      OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(ps.opModelName, ps.zeroPoint, ps.K, ps.N, ps.F0);

  std::vector<size_t> ifmShape;
  size_t ifmSize;
  if ((ps.zeroPoint == 29172) && (ps.opModelName == "m3uec")) {
    /* This is a specific case required for layer 1 of m3uec model only */
    ifmShape = {1, (static_cast<size_t>(ps.Mi0) * static_cast<size_t>(ps.Mi1)),
                static_cast<size_t>(ps.K + 1)};
    ifmSize = (ps.K + 1) * ps.Mi0 * ps.Mi1;
  } else if ((ps.zeroPoint == 40597) && ((ps.opModelName == "mswbjvw640") ||
                                         (ps.opModelName == "mswbjvw1280") ||
                                         (ps.opModelName == "mswbjvw2560"))) {
    /* This is a specific case required for layer 1 of PSO640 model only */
    ifmShape = {1, (static_cast<size_t>(ps.Mi0) * static_cast<size_t>(ps.Mi1)),
                static_cast<size_t>(ps.K - 4)}; // activate
    ifmSize = (ps.K - 4) * ps.Mi0 * ps.Mi1;
  } else {
    ifmShape = {1, (static_cast<size_t>(ps.Mi0) * static_cast<size_t>(ps.Mi1)),
                static_cast<size_t>(ps.K)};
    ifmSize = ps.K * ps.Mi0 * ps.Mi1;
  }
  std::string fileName = testDataFolder + "\\" + "ifm" + ".const";
  /* Sandip TBD : Need to replace vector type from opIfmDtype */
  std::vector<uint16_t> ifm = OpsFusion::read_bin_file<uint16_t>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }
  /* Add the ifm data to the ifmDataContainer */
  ifmDataContainer.push_back(std::move(ifm));
  /* Get a reference to the last element of the ifmDataContainer which contains
   * the ifm data just added */
  std::vector<uint16_t> &ifm_data_ref = ifmDataContainer.back();

  return {ifm_data_ref.data(), ifmShape, ps.opIfmDtype};
}

static bool GetOperateorHasWeights(std::string operatorName) {
  if ((operatorName == "maxpool") /* || (operatorName == "lstm") */) {
    return false;
  } else {
    return true;
  }
}

/* Helper function to read weight data from a file */
static std::vector<Tensor>
read_weight_data(const ParamsStruct &ps,
                 std::vector<std::vector<uint16_t>> &weightDataContainer) {
  if (!GetOperateorHasWeights(ps.operatorName)) {
    /* Return an empty Tensor with nullptr data, empty shape, and empty dtype */
    return {{nullptr, std::vector<size_t>{}, ""}};
  } else if (ps.operatorName == "lstm") {
    /* Add code to
            read 6 different files and return 6 Tensors */
    std::vector<Tensor> lstm_tensors;
    std::string testDataFolder = OpInterface::get_dd_base_dir() + "\\" +
                                 "tests" + "\\" + "cpp" + "\\" + "unit_tests" +
                                 "\\" + "testDataMladf" + "\\" + "lstm_" +
                                 std::to_string(ps.width);
    std::string fileName0W = testDataFolder + "\\" + "lstm_0_W_quantized.const";
    std::string fileName0R = testDataFolder + "\\" + "lstm_0_R_quantized.const";
    std::string fileName0B = testDataFolder + "\\" + "lstm_0_B_quantized.const";
    std::string fileName1W = testDataFolder + "\\" + "lstm_1_W_quantized.const";
    std::string fileName1R = testDataFolder + "\\" + "lstm_1_R_quantized.const";
    std::string fileName1B = testDataFolder + "\\" + "lstm_1_B_quantized.const";
    std::vector<std::string> lstm_filenames = {
        fileName0W, fileName0R, fileName0B, fileName1W, fileName1R, fileName1B};
    int fileCount = 0;
    for (const auto &file : lstm_filenames) {
      /* Read from file, similar to existing code for non-lstm case */
      // Fill in and adapt the code that reads weight data from a single file
      // into a Tensor
      std::vector<uint16_t> weight = OpsFusion::read_bin_file<uint16_t>(file);

      /* Add the weight data to the weightDataContainer */
      weightDataContainer.push_back(std::move(weight));
      /* Get a reference to the last element of the weightDataContainer which
       * contains the weight data just added */
      std::vector<uint16_t> &weight_data_ref = weightDataContainer.back();
      std::vector<size_t> weightShape;
      switch (fileCount) {
      case 0:
        weightShape = std::vector<size_t>{2, 512, 64};
        break;
      case 1:
      case 4:
        weightShape = std::vector<size_t>{2, 512, 128};
        break;
      case 2:
      case 5:
        weightShape = std::vector<size_t>{2, 1024};
        break;
      case 3:
        weightShape = std::vector<size_t>{2, 512, 256};
        break;
      default:
        break;
      }
      fileCount++;

      // Here, construct the Tensor for the current LSTM file using
      // weight_data_ref You may also need the shapes and dtypes for each file
      // data
      lstm_tensors.push_back(
          {weight_data_ref.data(), weightShape, ps.opWtsDtype});
    }
    return lstm_tensors;
  } else {
    /* Read weight data from a file */
    std::string opModelNameLowerCase = ps.opModelName;
    std::transform(opModelNameLowerCase.begin(), opModelNameLowerCase.end(),
                   opModelNameLowerCase.begin(), ::tolower);
    std::string testDataFolder =
        OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
        "unit_tests" + "\\" + "testDataMladf" + "\\" +
        GetTestSubFolderName(opModelNameLowerCase, ps.zeroPoint, ps.K, ps.N,
                             ps.F0);
    std::string fileName = testDataFolder + "\\" + "weight" + ".const";
    std::vector<size_t> weightShape;
    /* Sandip TBD: uint16_t is hardcoded. Should be modified with the use of
     * opWtsDtype */
    std::vector<uint16_t> weight = OpsFusion::read_bin_file<uint16_t>(fileName);

    if (ps.operatorName == "convformatmuladd") {
      weightShape = std::vector<size_t>{static_cast<size_t>(ps.K),
                                        static_cast<size_t>(ps.N)};
    } else {
      weightShape = (ps.groupId == 1)
                        ? std::vector<size_t>{static_cast<size_t>(ps.N),
                                              static_cast<size_t>(ps.K),
                                              static_cast<size_t>(ps.F0),
                                              static_cast<size_t>(ps.F1)}
                        : std::vector<size_t>{static_cast<size_t>(ps.N), 1,
                                              static_cast<size_t>(ps.F0),
                                              static_cast<size_t>(ps.F1)};

      if (weight.size() !=
          (weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3])) {
        std::cout << "Weight parameter file is not proper. Expected size = "
                  << (weightShape[0] * weightShape[1] * weightShape[2] *
                      weightShape[3])
                  << ", Actual Size = " << weight.size() << std::endl;
      }
    }
    /* Add the weight data to the weightDataContainer */
    weightDataContainer.push_back(std::move(weight));
    /* Get a reference to the last element of the weightDataContainer which
    contains the weight data just added */
    std::vector<uint16_t> &weight_data_ref = weightDataContainer.back();
    return {{weight_data_ref.data(), weightShape, ps.opWtsDtype}};
  }
}

/* Helper function to get attributes for each operator test is executing */
static std::map<std::string, std::any> GetAttr(const ParamsStruct &ps) {
  /* Store Attributes */
  std::map<std::string, std::any> attr;
  attr["opType"] = ps.operatorName;
  attr["opIfmDtype"] = ps.opIfmDtype;
  attr["opWtsDtype"] = ps.opWtsDtype;
  attr["opOfmDtype"] = ps.opOfmDtype;

  attr["group"] = std::vector<int>{ps.groupId};
  if (ps.operatorName == "convformatmuladd") {
    attr["input_shape"] = std::vector<int>{ps.Mi1, ps.Mi0, ps.K};
    attr["output_shape"] = std::vector<int>{ps.Mo1, ps.Mo0, ps.N};
  } else {
    attr["input_shape"] = std::vector<int>{1, ps.K, ps.Mi0, ps.Mi1};
    attr["output_shape"] = std::vector<int>{1, ps.N, ps.Mo0, ps.Mo1};
  }
  int weightThirdDim = (ps.groupId == 1) ? ps.K : 1;
  if ((ps.operatorName == "conv") || (ps.operatorName == "maxpool")) {
    attr["weight_shape"] = std::vector<int>{ps.N, weightThirdDim, ps.F0, ps.F1};
  } else if (ps.operatorName == "convformatmuladd") {
    attr["weight_shape"] = std::vector<int>{weightThirdDim, ps.N};
  }
  attr["zero_point"] = std::vector<int>{ps.zeroPoint};

  if (ps.opModelName != "mswbjvw") {
    /* Currently this unit test is desinged for mswbjvw only. Later this may
     * need a few changes based on requirement of the model*/
    attr["width"] = std::vector<int>{ps.width};
  }

  if ((ps.opModelName == "mswbjvw") && (ps.operatorName == "lstm")) {
    /* Sandip TBD: Below fix is to accomodate existing lstm. It is better that
     * all operators work with 4D input and output vectors */
    attr["input_shape"] = std::vector<int>{ps.Mi1, ps.K, ps.Mi0};
    attr["output_shape"] = std::vector<int>{ps.Mo1, ps.N, ps.Mo0};

    /* Sandip TBD: Below are workarounds only to test FC layer. Below parameters
     * are not actual values based on mswbjvw 320 model */
    attr["list_scale"] = std::vector<float>{
        0.00010169961751671508, 0.00008803531090961769, 0.00009158127795672044,
        0.00005283662903821096, 0.00003045230005227495, 0.00003045230005227495,
        0.00010079697676701471, 0.00008443013211945072, 0.00003503978223307058,
        0.000030487684853142127};
    attr["list_zero_point"] = std::vector<int>{
        25376, 26093, 36514, 31887, 32806, 32806, 28437, 33184, 25612, 32771};
  }

  return attr;
}

static std::string GetParamKey(std::string prefix, int64_t graphId,
                               int64_t inChannels, int64_t outChannels) {
  return prefix + "_" + std::to_string(graphId) + "_" +
         std::to_string(inChannels) + "_" + std::to_string(outChannels);
}

template <typename InT = uint16_t, typename OuT = uint16_t>
static int test_concatenate(
    const std::vector<std::tuple<int, int, int, int, int, int, int, int, int,
                                 bool, std::string, std::string, std::string,
                                 int, std::string, bool, int, std ::string>>
        &paramsVec,
    const int graphId = 320, const int ortIndex = 0, const int inChannels = 8,
    const int outChannels = 16, const std::string &modelName = "mswbjvw") {
  int total_err_count = 0;
  std::vector<std::map<std::string, std::any>> attributesVec;
  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;
  std::vector<Tensor> output_Tensor;
  std::vector<std::vector<uint16_t>> weightDataContainer;
  std::vector<std::vector<uint16_t>> ifmDataContainer;
  std::vector<std::vector<uint16_t>> ofmDataContainer;
  bool debugFlag = true;

  for (const auto &paramsTuple : paramsVec) {
    int err_count = 0;
    ParamsStruct ps = ExtractParameters(paramsTuple);
    attributesVec.push_back(GetAttr(ps));
    /* Call the helper function to read weight data with tuple as the argument
     */
    std::vector<Tensor> weight_data = read_weight_data(ps, weightDataContainer);

    // Insert all the Tensors returned from read_weight_data into const_Tensor
    const_Tensor.insert(const_Tensor.end(), weight_data.begin(),
                        weight_data.end());
  }

  if (!paramsVec.empty()) {
    const auto &first_params = ExtractParameters(paramsVec.front());
    const auto &last_params = ExtractParameters(paramsVec.back());

    input_Tensor.push_back(GetIfmTensor(first_params, ifmDataContainer));
    output_Tensor.push_back(GetOfmTensor(last_params, ofmDataContainer));
  }
  int dummyOrtIndex = 0;
  ryzenai::concateOps concatenate_ = ryzenai::concateOps<InT, OuT>(
      graphId, dummyOrtIndex, inChannels, outChannels, attributesVec);
  concatenate_.set_params(modelName, debugFlag);
  concatenate_.get_buffer_reqs(input_Tensor, output_Tensor);
  concatenate_.initialize_const_params(const_Tensor);

  std::string testDataFolder =
      OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetParamKey("concatenate", graphId, inChannels, outChannels);
  std::string weightGeneratedFolder =
      OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" + "GeneratedWeights";
  std::string fileName, generatedFileName;
  if (debugFlag) {
    fileName = testDataFolder + "\\" + "wtsRef.txt";
    generatedFileName =
        weightGeneratedFolder + "\\" +
        GetParamKey("wtsGenerated", graphId, inChannels, outChannels) + ".txt";
    if (CompareFileContents(fileName, generatedFileName)) {
      std::cout << "Error: the weight generated is not proper" << std::endl;
      total_err_count++;
    }
  }
#if 0
  /* Currently commented because we just want to limit testing for flatten weight generation */
  concatenate_.execute(input_Tensor, output_Tensor);
#if 0
  /* Many times the simulation team provides ofm data in txt format. One time below code is used to convert this in bin format */
  fileName = testDataFolder + "\\" + "ofmRef.txt";
  size_t dataSize = output_Tensor.at(0).shape[0] *
                    output_Tensor.at(0).shape[1] * output_Tensor.at(0).shape[2];
  uint16_t *dataPtr = (uint16_t *)malloc(dataSize * sizeof(uint16_t));
  readTxtFileHex<uint16_t>(fileName, dataPtr, dataSize * sizeof(uint16_t));
  generatedFileName = testDataFolder + "\\" + "ofmRef.bin";
  write_bin_file(generatedFileName, (char *)dataPtr,
                 output_Tensor.at(0).shape[0] * output_Tensor.at(0).shape[1] *
                     output_Tensor.at(0).shape[2] * 2);
#endif
  fileName = testDataFolder + "\\" + "ofmRef.bin";
  generatedFileName = weightGeneratedFolder + "\\" + "ofmOut.bin";
  write_bin_file(generatedFileName, (char *)output_Tensor.at(0).data,
                 output_Tensor.at(0).shape[0] * output_Tensor.at(0).shape[1] *
                     output_Tensor.at(0).shape[2] * 2);
  if (CompareFileContents(fileName, generatedFileName)) {
    std::cout << "Error: the ofm generated is not proper" << std::endl;
    total_err_count++;
  }
#endif
  return total_err_count;
}
#if 0
TEST(ConcatenateTesta16w16c16_, PsoTest1) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;
  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {
          std::make_tuple(60, 320, 3, 3, 8, 16, 64, 320, 1, false, "uint16",
                          "uint16", "uint16", 40597, "mswbjvw", true, 320,
                          "conv"), // layer1
          std::make_tuple(64, 320, 2, 2, 16, 16, 32, 160, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "mswbjvw", true, 320,
                          "maxpool"),

          std::make_tuple(32, 160, 3, 3, 16, 32, 32, 160, 1, false, "uint16",
                          "uint16", "uint16", 32705, "mswbjvw", true, 320,
                          "conv"), // layer2
          std::make_tuple(32, 160, 2, 2, 32, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "mswbjvw", true, 320,
                          "maxpool"),

          std::make_tuple(16, 80, 1, 1, 32, 16, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 36423, "mswbjvw", true, 320,
                          "conv"), // layer3
          std::make_tuple(16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 33409, "mswbjvw", true, 320, "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 128, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 29586, "mswbjvw", true, 320, "conv"),
          std::make_tuple(16, 80, 1, 1, 128, 16, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 25513, "mswbjvw", true, 320, "conv"),
          std::make_tuple(16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 31530, "mswbjvw", true, 320, "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 128, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 32591, "mswbjvw", true, 320,
                          "conv"), // layer8
          std::make_tuple(16, 80, 2, 1, 128, 128, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "mswbjvw", true, 320,
                          "maxpool"),

          std::make_tuple(8, 80, 1, 1, 128, 32, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 31990, "mswbjvw", true, 320, "conv"),
          std::make_tuple(8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 36064, "mswbjvw", true, 320, "conv"),
          std::make_tuple(8, 80, 1, 1, 48, 256, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 35326, "mswbjvw", true, 320, "conv"),
          std::make_tuple(8, 80, 1, 1, 256, 32, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 34702, "mswbjvw", true, 320, "conv"),
          std::make_tuple(8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 30051, "mswbjvw", true, 320, "conv"),
          std::make_tuple(8, 80, 1, 1, 48, 256, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 35719, "mswbjvw", true, 320,
                          "conv"), // layer14
          std::make_tuple(8, 80, 2, 1, 256, 256, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "mswbjvw", true, 320,
                          "maxpool"),

          std::make_tuple(4, 80, 1, 1, 256, 64, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 26536, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 22444, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 32234, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 512, 64, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33891, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33497, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 31960, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 512, 16, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33774, "mswbjvw", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 16, 2, 1, 128, 1, false, "uint16",
                          "uint16", "uint16", 33774, "mswbjvw", true, 320, "lstm"),
          std::make_tuple(1, 80, 1, 1, 256, 548, 1, 80, 1, false, "uint16",
                          "uint16", "uint16", 32771, "mswbjvw", true, 320,
                          "convForMatmulAdd")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 320, 28, 8, 16, "mswbjvw");
}

TEST(ConcatenateTesta16w16c16_, PsoTest2) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {std::make_tuple(60, 640, 3, 3, 8, 16, 32, 320, 1, false,
                                   "uint16", "uint16", "uint16", 40597,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(32, 320, 3, 3, 16, 32, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 32705,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 32, 16, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 36423,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(16, 160, 3, 3, 16, 32, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33409,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 32, 128, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 29586,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 128, 16, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 25513,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(16, 160, 3, 3, 16, 32, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 31530,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 32, 128, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 32591,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 128, 32, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 31990,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(8, 160, 3, 3, 32, 48, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 36064,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 48, 256, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 35326,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 256, 32, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 34702,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(8, 160, 3, 3, 32, 48, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 30051,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 48, 256, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 35719,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 256, 64, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 26536,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 3, 3, 64, 80, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 22444,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 80, 512, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 32234,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 512, 64, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33891,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 3, 3, 64, 80, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33497,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 80, 512, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 31960,
                                   "mswbjvw640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 512, 16, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33774,
                                   "mswbjvw640", true, 640, "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 640, 8, 16, "mswbjvw");
}

TEST(ConcatenateTesta16w16c16_, PsoTest3) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {std::make_tuple(60, 1280, 3, 3, 8, 16, 32, 640, 1, false,
                                   "uint16", "uint16", "uint16", 40597,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(32, 640, 3, 3, 16, 32, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 32705,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 32, 16, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 36423,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 3, 3, 16, 32, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33409,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 32, 128, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 29586,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 128, 16, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 25513,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 3, 3, 16, 32, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 31530,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 32, 128, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 32591,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 128, 32, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 31990,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 3, 3, 32, 48, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 36064,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 48, 256, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 35326,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 256, 32, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 34702,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 3, 3, 32, 48, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 30051,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 48, 256, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 35719,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 256, 64, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 26536,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 3, 3, 64, 80, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 22444,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 80, 512, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 32234,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 512, 64, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33891,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 3, 3, 64, 80, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33497,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 80, 512, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 31960,
                                   "mswbjvw1280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 512, 16, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33774,
                                   "mswbjvw1280", true, 1280, "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 1280, 8, 16, "mswbjvw");
}

TEST(ConcatenateTesta16w16c16_, PsoTest4) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {std::make_tuple(60, 2560, 3, 3, 8, 16, 32, 1280, 1, false,
                                   "uint16", "uint16", "uint16", 40597,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(32, 1280, 3, 3, 16, 32, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 32705,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 32, 16, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 36423,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 3, 3, 16, 32, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33409,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 32, 128, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 29586,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 128, 16, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 25513,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 3, 3, 16, 32, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 31530,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 32, 128, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 32591,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 128, 32, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 31990,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 3, 3, 32, 48, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 36064,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 48, 256, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 35326,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 256, 32, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 34702,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 3, 3, 32, 48, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 30051,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 48, 256, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 35719,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 256, 64, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 26536,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 3, 3, 64, 80, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 22444,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 80, 512, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 32234,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 512, 64, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33891,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 3, 3, 64, 80, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33497,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 80, 512, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 31960,
                                   "mswbjvw2560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 512, 16, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33774,
                                   "mswbjvw2560", true, 2560, "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 2560, 8, 16, "mswbjvw");
}
#endif
