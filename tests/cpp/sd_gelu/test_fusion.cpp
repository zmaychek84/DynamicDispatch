#include "test_common.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/gelu_e/gelue.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

template <typename InT = uint16_t, typename OuT = uint16_t>
static int test_gelu(const std::string &meta_json, size_t B, size_t M, size_t N,
                     bool debug = false,
                     const std::string &x_dtype = "bfloat16",
                     const std::string &y_dtype = "bfloat16") {

  std::vector<size_t> x_shape = {B, M, N};
  std::vector<size_t> y_shape = {B, M, N};

  std::vector<InT> x(B * M * N);
  std::vector<OuT> y(B * M * N, garbage_value);
  std::vector<float> y_golden(B * M * N, garbage_value);

  srand(42);
  dd::initialize_random_bfloat16(x, 4);

  const std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + "\\xclbin\\stx\\SD15_unet_2x4x4.xclbin";

  std::cout << xclbin_fname << "\n";
  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  cfg.model_name = "SD15";
  std::cout << "compile \n";
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");

  std::cout << "init \n";
  rt.init(meta, "", cfg);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{x.data(), x_shape, x_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{y.data(), y_shape, y_dtype}};

  std::cout << "execute \n";
  rt.execute(input_Tensors, output_Tensors);

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_gelu.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t B = 2;
  size_t M = 4096;
  size_t N = 1280;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_gelu(meta_json, B, M, N, false);
    if (err_count > 0) {
      std::cout << "Gelu test failed with err_count = " << err_count
                << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
