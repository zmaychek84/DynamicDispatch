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

#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/mladfmharope/mladfmharope.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"

// z = RoPE(x)
template <typename InT = uint16_t, typename TrigT = uint16_t,
          typename OuT = uint16_t>
static int test_mladfmharope(const std::string &meta_json, size_t B, size_t M,
                             size_t K, bool debug = false,
                             const std::string &a_dtype = "bfloat16",
                             const std::string &trig_dtype = "bfloat16",
                             const std::string &c_dtype = "bfloat16") {
  // TODO
  // start of duplicated code from unit test
  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> trig_shape = {2, M, K};

  // simple test vector for functionality
  // ifm = all ones
  // trig = all ones
  std::vector<InT> a(B * M * K, ryzenai::float_to_bfloat16(1.0f));
  std::vector<TrigT> trig(2 * M * K, ryzenai::float_to_bfloat16(1.0f));
  //  ==> Rope = half zeros half two
  // TODO: I believe these should be interleaved but current kernel has
  // contigous K/2 0s and then 2s
  std::vector<float> cpu_float(B * M * K, 0.0f);
  for (int i = 0; i < cpu_float.size(); ++i) {
    if (i % K >= K / 2) {
      cpu_float.at(i) = 2.0f;
    }
  }

  std::vector<OuT> aie_out(B * M * K, garbage_value);

  const auto &xclbin_fname =
      Utils::get_env_var("DD_ROOT") +
      ryzenai::
          LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;

  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype},
                   {trig.data(), trig_shape, trig_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{aie_out.data(), a_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return dd::count_errors_floatvsbfloat16(
      cpu_float, aie_out, a_shape,
      ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_mladfmharope.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }
  size_t B = 32;
  size_t M = 4096;
  size_t N = 128;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_mladfmharope(meta_json, B, M, N, false);
    if (err_count > 0) {
      std::cout << "mladfmharope test failed with err_count = " << err_count
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
