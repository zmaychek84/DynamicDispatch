/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <ops/sd/conv2d.hpp>
using namespace matmul_matrix;

std::vector<uint32_t> read_hex_file(const std::string &filePath) {
  std::ifstream fileStream(filePath);

  if (!fileStream.is_open()) {
    std::cerr << "Failed to open file " << filePath << "!" << std::endl;
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
void dump_data_as_uint32_to_file(const std::vector<T> &data,
                                 const std::string &output_file) {
  static_assert(sizeof(T) <= sizeof(uint32_t),
                "Data type is larger than uint32_t!");

  size_t num_uint32_elements = (data.size() * sizeof(T)) / sizeof(uint32_t);
  const uint32_t *data_as_uint32 =
      reinterpret_cast<const uint32_t *>(data.data());

  // Open the output file for writing
  std::ofstream ofm_ofs(output_file, std::ofstream::out | std::ofstream::trunc);
  if (!ofm_ofs.is_open()) {
    std::cerr << "Failed to open file " << output_file << " for writing!"
              << std::endl;
    return;
  }

  std::cout << "Opened file " << output_file << " for writing OFM!"
            << std::endl;

  for (size_t i = 0; i < num_uint32_elements; i++) {
    ofm_ofs << std::setw(8) << std::hex << std::setfill('0')
            << data_as_uint32[i] << std::endl;
  }
  ofm_ofs.close();
  std::cout << "Data has been successfully written to " << output_file
            << std::endl;
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_sd_conv(int IC, int IH, int IW, int OC, int OH, int OW, int kh, int kw,
                 int strideH, int strideW, bool debug = false,
                 const std::string &ifm_type = "bfloat16", // a bo
                 const std::string &wgt_type = "bfloat16", // b bo
                 const std::string &out_type = "bfloat16", // c bo
                 const std::string &model_name = "SD_VAE_DEC") {
  int err_count = 0;
  std::map<std::string, std::string> txnbin_a_header = {
      {"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  std::map<std::string, std::string> txnbin_b_header = {
      {"float", "w16bf"}, {"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  size_t ICs = static_cast<size_t>(IC);
  size_t IHs = static_cast<size_t>(IH);
  size_t IWs = static_cast<size_t>(IW);
  size_t OCs = static_cast<size_t>(OC);
  size_t OHs = static_cast<size_t>(OH);
  size_t OWs = static_cast<size_t>(OW);
  size_t khs = static_cast<size_t>(kh);
  size_t kws = static_cast<size_t>(kw);

  // first step, not populated
  std::vector<float> bias(OC);

  std::vector<size_t> a_shape = {1, IHs, IWs, ICs};
  std::vector<size_t> aie_out_shape = {1, OHs, OWs, OCs};
  std::vector<OuT> aie_out(OC * OH * OW);

  int wgt_size = 0;

  // std::string ifm_type;
  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{1, IH, IW, IC};
  attr["output_shape"] = std::vector<int>{1, OH, OW, OC};
  attr["weight_shape"] = std::vector<int>{OC, kh, kw, IC};
  const std::string bias_type = "float32";
  ryzenai::sd::conv sd_conv =
      ryzenai::sd::conv<std::uint16_t, std::uint8_t, float, std::uint16_t>(
          ifm_type, wgt_type, bias_type, out_type, false, attr);
  sd_conv.debug(debug);
  ryzenai::sd_conv2d_shapes shapes(OCs, ICs, IHs, IWs, OHs, OWs, khs, kws);
  sd_conv.set_params(model_name, shapes);

  std::string test_golden_root_dir =
      "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_conv/";
  std::string shape_key =
      txnbin_a_header.at(ifm_type) + txnbin_b_header.at(wgt_type) +
      txnbin_acc_header.at(out_type) + "_" + std::to_string(OC) + "_" +
      std::to_string(IC) + "_" + std::to_string(IH) + "_" + std::to_string(IW) +
      "_" + std::to_string(OH) + "_" + std::to_string(OW) + "_" +
      std::to_string(kh) + "_" + std::to_string(kw);
  std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
  std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

  std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
  std::vector<uint32_t> b_aie = read_hex_file(wts_path);
  std::vector<size_t> b_shape = {b_aie.size() * sizeof(uint32_t)};

  std::vector<Tensor> const_Tensor;
  // bias is actually not used because it is merged to b_aie.
  const_Tensor = {{b_aie.data(), b_shape, wgt_type}};

  sd_conv.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;

  input_Tensor = {{a_aie.data(), a_shape, ifm_type}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("OC = " << OC << ", OH = " << OH << ", OW = " << OW
                   << ", IC = " << IC << ", IH = " << IH << ", IW = " << IW);
  PROFILE_THIS(sd_conv.execute(input_Tensor, output_Tensor));
#else
  sd_conv.execute(input_Tensor, output_Tensor);
#endif

  std::string output_golden_path =
      test_golden_root_dir + shape_key + "_ofm32_ref.txt";
  std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
  auto output_size = OC * OH * OW;
  OutMatrix<OuT, 1, 1> output_golden_m(1, output_size, output_golden.data());
  OutMatrix<OuT, 1, 1> output_aie_m(1, output_size, aie_out.data());

  err_count = check_result_bfloat(output_golden_m, output_aie_m, true);

  std::string OFM_FILE = test_golden_root_dir + shape_key + "_aie_ofm32.txt";
  dump_data_as_uint32_to_file(aie_out, OFM_FILE);

  return err_count;
}

TEST(SD_CONV_Test, Kernel1) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      256, 512, 8, 128, 512, 8, 1, 1, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONV_Test, Kernel2) {
  int err_count = test_sd_conv<uint16_t, uint8_t, uint16_t>(
      128, 512, 512, 128, 512, 512, 3, 3, 1, 1, false, "bfloat16", "bfp16ebs8",
      "bfloat16", "SD_VAE_DEC");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
