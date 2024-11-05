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

#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

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

int test_sd_conv(const std::string &meta_json) {
  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + "\\xclbin\\stx\\SDConv2d.xclbin";
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");
  std::cerr << "Compiled" << std::endl;

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");
  rt.init(meta);

  std::string test_golden_root_dir =
      "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_conv/";

  std::string ifm_path = test_golden_root_dir +
                         "a16bfw16bfpacc16bf_128_256_512_8_512_8_1_1_ifm32.txt";
  std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

  const size_t IH = 512;
  const size_t IW = 8;
  const size_t IC = 256;
  const size_t OH = 512;
  const size_t OW = 8;
  const size_t OC = 128;

  std::vector<size_t> a_shape = {1, IH, IW, IC}; // nhwc
  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a_aie.data(), a_shape, "bfloat16"}};

  std::vector<size_t> aie_out_shape = {1, OH, OW, OC};
  std::vector<uint16_t> aie_out(OH * OW * OC, 0);
  std::vector<Tensor> output_Tensors;
  struct Tensor c_T = {aie_out.data(), aie_out_shape, "bfloat16"};
  output_Tensors.push_back(c_T);
  rt.execute(input_Tensors, output_Tensors);
  // std::string OFM_FILE = "aie_ofm32_fusiont.txt";
  // dump_data_as_uint32_to_file(aie_out, OFM_FILE);

  std::string output_golden_path =
      test_golden_root_dir +
      "a16bfw16bfpacc16bf_128_256_512_8_512_8_1_1_ofm32_ref.txt";
  std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
  auto output_size = OC * OH * OW;
  OutMatrix<uint16_t, 1, 1> output_golden_m(1, output_size,
                                            output_golden.data());
  OutMatrix<uint16_t, 1, 1> output_aie_m(1, output_size, aie_out.data());

  return check_result_bfloat(output_golden_m, output_aie_m, true);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_sd_conv.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_sd_conv(meta_json);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
