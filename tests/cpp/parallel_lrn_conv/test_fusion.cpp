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
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"

static void test_model(const std::string &meta_json, size_t niters) {
  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") +
      "xclbin\\stx\\4x2_psi_integrated_model_a16w8_qdq.xclbin";
  auto meta = OpsFusion::load_meta_json(meta_json);

  auto xrt_context =
      ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin_fname);
  auto context = xrt_context->get_context();

  OpsFusion::DDConfig cfg = {3, false};
  OpsFusion::FusionRuntime rt(&context);
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  rt.compile(meta, "", cfg);
  rt.init(meta);
  auto fops2 = rt.get_txns();

  // Prepare inputs
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<Tensor> in_tensors =
      OpsFusion::MetaUtils::get_input_tensors(meta);
  for (auto &tensor : in_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);
    std::vector<uint8_t> in(sz, 1);
    rand_init_int((int8_t *)in.data(), in.size() / sizeof(int8_t));
    inputs.push_back(std::move(in));
    tensor.data = inputs.back().data();
  }

  // Prepare outputs
  std::vector<Tensor> out_tensors =
      OpsFusion::MetaUtils::get_output_tensors(meta);
  std::vector<std::vector<uint8_t>> outputs;
  for (auto &tensor : out_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);

    outputs.emplace_back(sz, 1);
    tensor.data = outputs.back().data();
  }

  std::cout << OpsFusion::MetaUtils::get_summary(rt.get_meta()) << std::endl;

  std::cout << "Executing for iterations:" << niters << std::endl;
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < niters; ++i) {
    rt.execute(in_tensors, out_tensors);
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "Avg. Time (ms) : "
            << std::chrono::duration<float, std::milli>(t2 - t1).count() /
                   niters
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json> [niters=1]" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    size_t niters = (argc > 2) ? std::atoll(argv[2]) : 1;
    test_model(meta_json, niters);

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
