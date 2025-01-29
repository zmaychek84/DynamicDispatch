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

static std::vector<char> load_bin(const std::string &filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Couldn't open file : "s + filename);
  }

  std::istreambuf_iterator<char> begin_it{ifs}, end_it;
  std::vector<char> data(begin_it, end_it);
  return data;
}

static int max_abs_error(const int16_t *vec1, const int16_t *vec2,
                         size_t size) {
  int err = 0;
  for (size_t i = 0; i < size; ++i) {
    err = std::max(err, std::abs(vec1[i] - vec2[i]));
  }
  return err;
}

using MatrixShape = std::array<size_t, 2>;
static void matmul_ref(const int16_t *A, const int8_t *B, int16_t *C,
                       MatrixShape A_shape, MatrixShape B_shape) {
  auto M = A_shape[0];
  auto K = A_shape[1];
  auto N = B_shape[1];
  if (K != B_shape[0]) {
    throw std::runtime_error("Matmul : Shape mismatch in inner dimension");
  }

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc{0};
      for (size_t k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(A[m * K + k]) *
               static_cast<int32_t>(B[k * N + n]);
      }
      C[m * N + n] = static_cast<int16_t>(
          std::clamp<int32_t>(acc, std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()));
    }
  }
}

static float vec_sum(const std::vector<uint8_t> &out2) {
  int16_t *ptr = (int16_t *)out2.data();
  size_t size = out2.size() / sizeof(int16_t);
  return std::accumulate(ptr, ptr + size, 0.0f,
                         [](float a, int16_t b) { return a + b; });
}

static void cpu_ref(const OpsFusion::Metadata &meta,
                    const std::vector<void *> &ins,
                    const std::vector<void *> &outs, size_t M, size_t K,
                    size_t N) {
  // Weights
  std::vector<std::vector<char>> weights;
  for (auto &[tname, tinfo] : meta.tensor_map) {
    if (tinfo.parent_name != "const") {
      continue;
    }
    std::vector<char> data = load_bin(tinfo.file_name);

    std::cout << "Load const : " << tinfo.file_name << ", " << data.size()
              << ", " << tinfo.file_size << std::endl;
    if (data.size() != tinfo.file_size) {
      throw std::runtime_error("Size of file:"s + tinfo.file_name +
                               " doesn't match with metadata info.");
    }
    weights.push_back(std::move(data));
  }
  int16_t *X = reinterpret_cast<int16_t *>(ins.front());
  int16_t *Y = reinterpret_cast<int16_t *>(outs.front());

  MatrixShape act_shape{M, K};
  MatrixShape wts_shape{K, N};

  matmul_ref(X, reinterpret_cast<int8_t *>(weights.at(0).data()), Y, act_shape,
             wts_shape);
}

struct test_model_flags {
  bool load_only;
  bool save_only;
  bool write_input;
  bool write_output;
  bool compare_output;
  bool no_execute;

  test_model_flags()
      : load_only(false), save_only(false), write_input(false),
        write_output(false), compare_output(false), no_execute(false) {}
};

static void test_model_via_json() {}

static void test_model_via_state() {}

static void test_model(const std::string &meta_json,
                       const std::string &state_filename,
                       const std::string &xclbin_fname, int niters,
                       const test_model_flags *flags) {

  auto meta = OpsFusion::load_meta_json(meta_json);
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content, "DPU");

  // if state file not specified, create dd_metastate from the meta and load it.
  // unless load_only was specified, in which case load dd_metastate
  // if state file is specified, initialize the RT with that
  if (state_filename.empty()) {
    if (!flags->load_only) {
      OpsFusion::DDConfig cfg;
      cfg.xclbin_content = &xclbin_content;
      OpsFusion::FusionRuntime rt_cmp;
      rt_cmp.compile(meta, "", cfg);
      rt_cmp.save_state("dd_metastate");
      std::cerr << "dd_metastate saved" << std::endl;
    }
    std::cerr << "FusionRuntime initializing from dd_metastate" << std::endl;
    rt.load_state("dd_metastate");
    rt.init(meta);
    std::cerr << "FusionRuntime initialized from dd_metastate" << std::endl;
  } else {
    rt.load_state(state_filename);
    rt.init(meta);
    std::cerr << "FusionRuntime initialized from " << state_filename.c_str()
              << std::endl;
  }

  if (flags->save_only) {
    return;
  }

  // Prepare inputs
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<Tensor> in_tensors =
      OpsFusion::MetaUtils::get_input_tensors(meta);
  int input_idx = 0;
  for (auto &tensor : in_tensors) {
    ++input_idx;
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);
    std::vector<uint8_t> in(sz, 1);
    rand_init_int((int8_t *)in.data(), in.size() / sizeof(int8_t));
    inputs.push_back(std::move(in));
    tensor.data = inputs.back().data();

    if (flags->write_input) {
      std::string ofname = "input-" + std::to_string(input_idx) + ".bin";
      std::ofstream ofs(ofname, std::ios::binary);
      ofs.write((const char *)tensor.data, sz);
      std::cout << "wrote to file " << ofname.c_str() << " from address "
                << (uintptr_t)tensor.data << " "
                << "of size " << sz << " " << std::endl;
    }
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

  if (flags->no_execute) {
    if (flags->write_output || flags->compare_output) {
      std::cerr << "Warning: the X flag (no_execute) flag means the O and C "
                   "flags (write output, compare output) are ignored."
                << std::endl;
    }
    return;
  }

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

  if (flags->write_output || flags->compare_output) {
    int output_idx = 0;
    for (auto &tensor : out_tensors) {
      ++output_idx;
      size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                  size_t{1}, std::multiplies{}) *
                  Utils::get_size_of_type(tensor.dtype);

      if (flags->write_output) {
        std::string ofname = "output-" + std::to_string(output_idx) + ".bin";
        std::ofstream ofs(ofname, std::ios::binary);
        ofs.write((const char *)tensor.data, sz);
        std::cout << "wrote " << ofname.c_str() << " from " << std::hex
                  << (uintptr_t)tensor.data << " size " << std::dec << sz << " "
                  << std::endl;
      }

      if (flags->compare_output) {
        std::string ifname = "golden-" + std::to_string(output_idx) + ".bin";
        std::ifstream ifs(ifname, std::ios::binary);
        std::vector<char> ibuf;
        ibuf.resize(sz);
        ifs.read(ibuf.data(), sz);
        int rv = memcmp(tensor.data, ibuf.data(), sz);
        std::cout << "compare " << ifname.c_str() << " from " << std::hex
                  << (uintptr_t)tensor.data << " size " << std::dec << sz
                  << " : " << (rv ? "FAIL" : "PASS") << " " << std::endl;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage : " << argv[0] << " "
              << "meta.json meta.state file.xclbin {L,S} [I,O,C,X] [niters=1]"
              << std::endl
              << "L = load_only, do not save" << std::endl
              << "S = save_only, do not contine" << std::endl
              << "I = write_input, to files named \"input-#\".bin" << std::endl
              << "O = write_output, to files named \"output-#\".bin"
              << std::endl
              << "C = compare_output, to files named \"golden-#\".bin"
              << std::endl
              << "X = no_execute, terminate before execution" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json_filename = std::string(argv[1]);
    std::string meta_state_filename = std::string(argv[2]);
    std::string xclbin_filename =
        Utils::get_env_var("DD_ROOT") + "/xclbin/stx/" + argv[3];
    int niters = 1;
    test_model_flags flags;

    for (int i = 4; i < argc; ++i) {
      switch (argv[i][0]) {
      case 'L':
        flags.load_only = true;
        break;
      case 'S':
        flags.save_only = true;
        break;
      case 'I':
        flags.write_input = true;
        break;
      case 'O':
        flags.write_output = true;
        break;
      case 'C':
        flags.compare_output = true;
        break;
      case 'X':
        flags.no_execute = true;
        break;
      default:
        niters = std::atoll(argv[i]);
      }
    }

    test_model(meta_json_filename, meta_state_filename, xclbin_filename, niters,
               &flags);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
