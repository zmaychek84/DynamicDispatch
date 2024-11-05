#include "test_common.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/gelu_e/gelue.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

// gelu is computed as x*sigmoid(1.702x)
static inline float sigmoid(float xf) { return 1.0f / (1.0f + std::exp(-xf)); }

// gelu has this polynomial expression for  x*sigmoid(1.702x)
// this is often the backbone of the AIE gelu core computation
static inline float g(float x) {
  float p1 = 1.702f;
  float q0 = 1 / 2;
  float q1 = 0.25f;
  float q2 = 0.03125f;
  float mul1, mul1_abs, mul0, mac0, msc, mul2;

  mul1 = p1 * x;

  // sigmoid
  mul1_abs = (mul1 > 0) ? mul1 : -mul1;

  mul0 = q2 * mul1_abs;
  mac0 = q0 + mul1 * q1;
  msc = mac0 - mul0 * mul1;
  mul2 = msc * x;
  return mul2;
}

// This is the erf approximation
static inline float erf_p(float x) {
  // constants
  float a1 = 0.254829592f;
  float a2 = -0.284496736f;
  float a3 = 1.421413741f;
  float a4 = -1.453152027f;
  float a5 = 1.061405429f;
  float p = 0.3275911f;

  // Save the sign of x
  int sign = 1;
  if (x < 0) {
    sign = -1;
  }
  x = fabs(x);

  // A&S formula 7.1.26
  float t = (float)1.0f / (1.0f + p * x);
  float y =
      1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

  return sign * y;
}

// This is the erf approximation
static inline float gdef(float x) {
  float sqrt2 = 1.4142135623730951f; // sqrt(2)
  return x * (1 + erf_p(x / sqrt2)) / 2;
}

// y = x * sigmoid(x)
template <typename InT = uint16_t, typename OuT = uint16_t>
static int test_gelu(const std::string &meta_json, size_t M, size_t N,
                     bool debug = false,
                     const std::string &x_dtype = "bfloat16",
                     const std::string &y_dtype = "bfloat16") {

  std::vector<size_t> x_shape = {1, M, N};
  std::vector<size_t> y_shape = {1, M, N};

  std::vector<InT> x(M * N);
  std::vector<OuT> y(M * N, garbage_value);
  std::vector<float> y_golden(M * N, garbage_value);

  srand(42);
  dd::initialize_random_bfloat16(x, 4);

  // compute golden
  for (int i = 0; i < M * N; ++i) {
    float xfte = ryzenai::bfloat16_to_float(x[i]);
    // This is not the golden
    // the definition is x*(1+special.erf(x/math.sqrt(2)))/2

    y_golden[i] = gdef(xfte);
  }

  const std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") +
      "xclbin/stx/gelue_4x4_abfloat16cbfloat.xclbin";

  std::cout << xclbin_fname << "\n";
  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  std::cout << "compile \n";
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");

  std::cout << "init \n";
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{x.data(), x_shape, x_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{y.data(), y_shape, y_dtype}};

  std::cout << "execute \n";
  rt.execute(input_Tensors, output_Tensors);

  std::cout << "compare \n";
  return dd::count_errors_floatvsbfloat16(
      y_golden, y, y_shape, ryzenai::gelue<uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_gelu.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 1;
  size_t N = 11008;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_gelu(meta_json, M, N, false);
    if (err_count > 0) {
      std::cout << "Silu test failed with err_count = " << err_count
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
