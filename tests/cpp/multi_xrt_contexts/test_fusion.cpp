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

#include <iostream>
#include <memory>

#include <op_fuser/fusion_rt.hpp>
#include <ops/op_interface.hpp>

int create_xrt_contexts(std::uint32_t num_xrt_contexts) {
  std::cout << "Create num_xrt_contexts" << num_xrt_contexts << std::endl;

  auto xclbin = OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu.xclbin";

  auto DPU_KERNEL_NAME = "DPU";

  bool status = true;

  try {
    std::vector<std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context>> vec;

    for (std::uint32_t context_id = 0; context_id < num_xrt_contexts;
         context_id++) {
      std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> xrt_context =
          ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin,
                                                               context_id);
      vec.push_back(xrt_context);
    }

  } catch (...) {
    status = false;
  }

  if (!status) {
    std::cout << "create_xrt_contexts::Test Fail !" << std::endl;
    return -1;
  }

  std::cout << "create_xrt_contexts::Test Pass !" << std::endl;

  return 0;
}

int create_destroy_xrt_contexts(std::uint32_t num_xrt_contexts) {
  std::cout << "Create/destory num_xrt_contexts" << num_xrt_contexts
            << std::endl;

  auto xclbin = OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu.xclbin";

  auto DPU_KERNEL_NAME = "DPU";

  bool status = true;

  try {
    std::vector<std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context>> vec;

    for (std::uint32_t context_id = 0;
         context_id < std::min(ryzenai::dynamic_dispatch::MAX_NUM_XRT_CONTEXTS,
                               num_xrt_contexts);
         context_id++) {
      std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> xrt_context =
          ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin,
                                                               context_id);
      vec.push_back(xrt_context);
    }

    for (std::uint32_t context_id = std::min(
             ryzenai::dynamic_dispatch::MAX_NUM_XRT_CONTEXTS, num_xrt_contexts);
         context_id < num_xrt_contexts; context_id++) {
      vec.pop_back();
      std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> xrt_context =
          ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin,
                                                               context_id);
      vec.push_back(xrt_context);
    }

  } catch (...) {
    status = false;
  }

  if (!status) {
    std::cout << "create_destroy_xrt_contexts::Test Fail !" << std::endl;
    return -1;
  }

  std::cout << "create_destroy_xrt_contexts::Test Pass !" << std::endl;

  return 0;
}

int main() {

  std::vector<std::uint32_t> pass_num_contexts_cases = {
      1, 2, 4, 8, ryzenai::dynamic_dispatch::MAX_NUM_XRT_CONTEXTS};

  for (auto num_contexts : pass_num_contexts_cases) {
    int error_code = create_xrt_contexts(num_contexts);

    if (error_code != 0) {
      std::cout << "Expect " << num_contexts << " context to pass" << std::endl;
      return error_code;
    }
  }

#ifdef LIMITED_XRT_CONTEXT_TEST_EN
  std::vector<std::uint32_t> fail_num_contexts_cases = {
      ryzenai::dynamic_dispatch::MAX_NUM_XRT_CONTEXTS + 2,
      ryzenai::dynamic_dispatch::MAX_NUM_XRT_CONTEXTS + 4};

  for (auto num_contexts : fail_num_contexts_cases) {
    int error_code = create_xrt_contexts(num_contexts);

    if (error_code == 0) {
      std::cout << "Expect " << num_contexts << " context to fail" << std::endl;
      return -1;
    }
  }
#endif

  int error_code = create_destroy_xrt_contexts(
      ryzenai::dynamic_dispatch::MAX_NUM_XRT_CONTEXTS + 4);

  if (error_code != 0) {
    std::cout << "Expect create/destroy context to pass" << std::endl;
    return error_code;
  }

  return 0;
}
