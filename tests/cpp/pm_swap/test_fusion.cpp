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

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

#include <fstream>

#include "test_common.hpp"

using namespace std;

int test_pm_swap(const std::string &meta_json, size_t M, size_t K,
                 uint32_t case_idx) {

  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> b_shape = {M, K};

  uint32_t num_outputs = case_idx == 3 ? 2 : 1;

  std::vector<std::int32_t> a(M * K);
  std::vector<std::int32_t> cpu_out(num_outputs * M * K);
  std::vector<std::int32_t> aie_out(M * K, garbage_value);
  std::vector<std::int32_t> aie_out_2(M * K, garbage_value);

  initialize_random<std::int32_t>(a, M * K, 10, 0);
  // for (int i = 0; i < a.size(); i++) {
  //   a.at(i) = 12;
  // }

  auto square_function = [](std::int32_t val) { return val * val; };
  auto cube_function = [](std::int32_t val) { return val * val * val; };

  // compute golden
  if (0 == case_idx) {
    std::transform(a.begin(), a.end(), cpu_out.begin(), square_function);
  } else if (1 == case_idx) {
    std::transform(a.begin(), a.end(), cpu_out.begin(), cube_function);
  } else if (2 == case_idx) {
    std::transform(a.begin(), a.end(), cpu_out.begin(), square_function);
    std::transform(cpu_out.begin(), cpu_out.end(), cpu_out.begin(),
                   cube_function);
  } else if (3 == case_idx) {
    // 2 outputs
    std::transform(a.begin(), a.end(), cpu_out.begin(), square_function);
    std::transform(a.begin(), a.end(), cpu_out.begin() + cpu_out.size() / 2,
                   cube_function);
    // assumes layout is [x^4 | x^5]
    std::transform(cpu_out.begin(), cpu_out.end(), cpu_out.begin(),
                   square_function);
  }

  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + "xclbin\\stx\\square_cube.xclbin";

  auto meta = OpsFusion::load_meta_json(meta_json);
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  OpsFusion::DDConfig cfg = {3, true};
  rt.compile(meta, "", cfg);
  rt.init(meta, "", cfg);

  struct Tensor a_T = {a.data(), a_shape, "int32"};
  struct Tensor c_T = {aie_out.data(), a_shape, "int32"};
  struct Tensor c_T_2 = {aie_out_2.data(), a_shape, "int32"};

  std::vector<Tensor> input_Tensor;
  input_Tensor.push_back(a_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);
  if (3 == case_idx) {
    output_Tensor.push_back(c_T_2);
  }

  rt.execute(input_Tensor, output_Tensor);

  int err_count = 0;

  std::int32_t max_error = std::numeric_limits<std::int32_t>::min();
  if (3 != case_idx) {
    for (int i = 0; i < cpu_out.size(); ++i) {
      max_error = std::max(max_error, std::abs(cpu_out[i] - aie_out[i]));
      std::cout << cpu_out.at(i) << ", " << aie_out.at(i) << std::endl;
      if (cpu_out[i] != aie_out[i]) {
        err_count++;
      }
    }
  } else {

    std::cout << "First output" << std::endl;
    std::vector<int32_t> first_golden(M * K);
    for (int i = 0; i < a.size(); i++) {
      first_golden.at(i) = std::pow(a.at(i), 2);
      first_golden.at(i) = std::pow(first_golden.at(i), 3);
    }
    for (int i = 0; i < first_golden.size(); ++i) {
      max_error = std::max(max_error, std::abs(first_golden[i] - aie_out[i]));
      // std::cout << a.at(i) << ", " << first_golden.at(i) << ", " <<
      // aie_out.at(i) << std::endl;
      if (first_golden[i] != aie_out[i]) {
        err_count++;
      }
    }

    std::cout << "Second output" << std::endl;
    std::vector<int32_t> second_golden(M * K);
    for (int i = 0; i < a.size(); i++) {
      second_golden.at(i) = std::pow(a.at(i), 2);
      second_golden.at(i) = std::pow(second_golden.at(i), 2);
    }
    for (int i = 0; i < second_golden.size(); ++i) {
      max_error =
          std::max(max_error, std::abs(second_golden[i] - aie_out_2[i]));
      // std::cout << a.at(i) << ", " <<second_golden.at(i) << ", " <<
      // aie_out_2.at(i) << std::endl;
      if (second_golden[i] != aie_out_2[i]) {
        err_count++;
      }
    }
  }

  std::cout << "Max error : " << max_error << std::endl;

  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;

  try {
    std::string meta_json = std::string(argv[1]);

    int err_count = 0;

    if (meta_json.find("0") != string::npos) {
      err_count = test_pm_swap(meta_json, 1, 32, 0);
    } else if (meta_json.find("1") != string::npos) {
      err_count = test_pm_swap(meta_json, 1, 32, 1);
    } else if (meta_json.find("2") != string::npos) {
      err_count = test_pm_swap(meta_json, 1, 32, 2);
    } else if (meta_json.find("3") != string::npos) {
      err_count = test_pm_swap(meta_json, 1, 32, 3);
    } else {
      std::cout << "Unexpected test case!" << std::endl;
    }

    if (err_count > 1) {
      std::cout << "PM Test failed with err count : " << err_count << std::endl;
      return EXIT_FAILURE;
    } else {
      std::cout << "PM Test Passed with err count : " << err_count << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
