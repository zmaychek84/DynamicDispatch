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

struct test_model_flags {
  bool load_only;
  bool save_only;
  bool write_input;
  bool write_output;
  bool gen_golden;
  bool gen_state;
  bool compare_output;
  bool no_execute;
  bool no_avoid;
  bool local_xclbin;
  bool test_configs;
  bool print_summary;
  bool print_perf;
  bool print_debug;
  bool be_quiet;
  bool cleanup;
  uint32_t configs_to_test;
  uint32_t init_method;
  std::string json_filename;
  std::string state_filename;
  std::string xclbin_filename;

  test_model_flags()
      : load_only(false), save_only(false), write_input(false),
        write_output(false), gen_golden(false), gen_state(false),
        compare_output(false), no_execute(false), no_avoid(false),
        local_xclbin(false), test_configs(false), print_summary(false),
        print_perf(false), print_debug(false), be_quiet(false), cleanup(false),
        configs_to_test(0), init_method(0) {}

  bool has_action() {
    return write_input || write_output || gen_golden || compare_output;
  }

  template <class T>
  void print(std::ostream &ostr, const char *var_name, T &var_val) const {
    ostr << "   " << var_name << " = " << var_val << " " << std::endl;
  }

  void dump(std::ostream &ostr) const {
    print(ostr, "load_only       = ", load_only);
    print(ostr, "save_only       = ", save_only);
    print(ostr, "write_input     = ", write_input);
    print(ostr, "write_output    = ", write_output);
    print(ostr, "gen_golden      = ", gen_golden);
    print(ostr, "gen_state       = ", gen_state);
    print(ostr, "compare_output  = ", compare_output);
    print(ostr, "no_execute      = ", no_execute);
    print(ostr, "no_avoid        = ", no_avoid);
    print(ostr, "local_xclbin    = ", local_xclbin);
    print(ostr, "test_configs    = ", test_configs);
    print(ostr, "configs_to_test = ", configs_to_test);
    print(ostr, "init_method     = ", init_method);
    print(ostr, "print_summary   = ", print_summary);
    print(ostr, "print_perf      = ", print_perf);
    print(ostr, "print_debug     = ", print_debug);
    print(ostr, "be_quiet        = ", be_quiet);
    print(ostr, "cleanup         = ", cleanup);
    print(ostr, "json_filename   = ", json_filename);
    print(ostr, "state_filename  = ", state_filename);
    print(ostr, "xclbin_filename = ", xclbin_filename);
  }

  // clang-format off

  void cleanup_other_files(std::string filename) const {
    if (cleanup) {
      if (!be_quiet) std::cout << "cleanup: ";
      { std::filesystem::path p(filename); p.replace_extension("state"   );  if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p); }
      { std::filesystem::path p(filename); p.replace_extension("fconst"  );  if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p); }
      { std::filesystem::path p(filename); p.replace_extension("super"   );  if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p); }
      { std::filesystem::path p(filename); p.replace_extension("input"   );  if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p); }
      { std::filesystem::path p(filename); p.replace_extension("ctrlpkt" );  if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p); }
      if (true) {
        if (!write_output) {
          for (int idx = 1; true; ++idx) {
            std::string ext = std::string("output-") + std::to_string(idx) + std::string(".bin");
            std::string ofname = std::filesystem::path(filename).replace_extension(ext).string();
            std::filesystem::path p(ofname);
            if (!std::filesystem::is_regular_file(p))
              break;
            if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p);
          }
        }
        if (!write_input) {
          for (int idx = 1; true; ++idx) {
            std::string ext = std::string("ddinput-") + std::to_string(idx) + std::string(".bin");
            std::string ofname = std::filesystem::path(filename).replace_extension(ext).string();
            std::filesystem::path p(ofname);
            if (!std::filesystem::is_regular_file(p))
              break;
            if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p);
          }
        }
        if (!gen_golden) {
          for (int idx = 1; true; ++idx) {
            std::string ext = std::string("golden-") + std::to_string(idx) + std::string(".bin");
            std::string ofname = std::filesystem::path(filename).replace_extension(ext).string();
            std::filesystem::path p(ofname);
            if (!std::filesystem::is_regular_file(p))
              break;
            if (!be_quiet) std::cout << p.string() << " ";  std::filesystem::remove(p);
          }
        }
      }
      if (!be_quiet) std::cout << std::endl;
    }
  }

  // clang-format on

  void dump() const {
    if (print_debug) {
      dump(std::cout);
    }
  }

  void qdump() const {
    if (!be_quiet) {
      dump(std::cout);
    }
  }

  template <typename A1, typename A2, typename A3>
  void dbgprint(A1 a1, A2 a2, A3 a3) const {
    if (print_debug) {
      std::cout << a1 << a2 << a3 << std::endl;
    }
  }

  template <typename A1, typename A2> void qprint(A1 a1, A2 a2) const {
    if (!be_quiet) {
      std::cout << a1 << a2 << std::endl;
    }
  }

  template <typename A1, typename A2, typename A3, typename A4>
  void qprint(A1 a1, A2 a2, A3 a3, A4 a4) const {
    if (!be_quiet) {
      std::cout << a1 << a2 << a3 << a4 << std::endl;
    }
  }
};

namespace TMF {
#include "tmf_strings.cc"
} // namespace TMF

const char *verify_is_a_regular_file(std::string &filename) {
  std::filesystem::path p(filename);
  if (!std::filesystem::exists(p)) {
    return "a file with that name does not exist";
  }
  if (std::filesystem::is_directory(p)) {
    return "the file name is actually a directory";
  }
  if (!std::filesystem::is_regular_file(p)) {
    return "the file with that name is not a regular file";
  }
  return nullptr;
}

bool does_avoid_file_exist(std::string &filename) {
  std::filesystem::path p(filename);
  std::string d = (p.has_parent_path() ? p.parent_path().string() + "/" : "") +
                  TMF::string_avoid;
  std::filesystem::path pd(d);
  bool it_does = std::filesystem::is_regular_file(pd);
  return it_does;
}

// forward declaration, called by test_model, calls test_model
static bool test_model_from_ddconfigs(const test_model_flags &flags, int niters,
                                      OpsFusion::Metadata meta,
                                      std::vector<char> &xclbin_content);

static bool test_model(const test_model_flags &flags, int niters) {
  if (flags.json_filename.empty() && flags.state_filename.empty()) {
    std::cout
        << "NOTE: state_filename and json_filename are empty, nothing to do."
        << std::endl;
    return false;
  }

  bool using_dd_metastate_file = false;
  auto xclbin_content = OpsFusion::read_bin_file<char>(flags.xclbin_filename);

  std::string state_filename_to_use;
  if (!flags.state_filename.empty()) {
    state_filename_to_use = flags.state_filename;

    if (flags.test_configs) {
      std::cout << "WARNING: using different DDConfigs is not supported when "
                   "a state file is specified."
                << std::endl;
    }
  } else {
    state_filename_to_use = TMF::string_dd_metastate + std::string(".state");
    using_dd_metastate_file = true;

    // when state file not given, use json and call save_state()
    // json is used only when the state file is not given
    flags.dbgprint("    ", flags.json_filename, " loading to meta");
    auto meta = OpsFusion::load_meta_json(flags.json_filename);

    // create and save dd_metastate
    if (!flags.load_only) {
      // only create and save dd_metastate when load_only is not set

      if (flags.test_configs) {
        return test_model_from_ddconfigs(flags, niters, meta, xclbin_content);
      }

      // save default dd_metastate.state files
      if (!flags.no_execute) {
        OpsFusion::DDConfig cfg;
        cfg.xclbin_content = &xclbin_content;
        OpsFusion::FusionRuntime rt_cmp;

        std::map<std::string, OpsFusion::SimpleSpan> const_map;
        std::map<std::string, std::vector<char>> const_buffers;
        std::filesystem::path path_to_cache(flags.json_filename);
        OpsFusion::FusionRuntime::build_const_map(
            meta, const_map, const_buffers,
            path_to_cache.replace_filename("").string());
        rt_cmp.compile(meta, "", cfg, const_map);
        rt_cmp.save_state(state_filename_to_use);
      }
    }
  }

  if (!flags.no_avoid) {
    if (does_avoid_file_exist(state_filename_to_use)) {
      return false;
    }
  }

  const std::filesystem::path pf(state_filename_to_use);

  if (!std::filesystem::exists(pf)) {
    // return if the given, or saved, state file does not exist
    if (!flags.state_filename.empty()) {
      // return fail if the state file was given
      std::cout << "FAIL"
                << " "
                << "does not exist: " << state_filename_to_use << " "
                << std::endl;
      return true;
    }
    // don't return fail, maybe the default state file was not created
    if (using_dd_metastate_file) {
      flags.cleanup_other_files(std::string(TMF::string_dd_metastate) +
                                ".state");
    }
    return false;
  }

  // do not return before the rt object is fully initialized
  OpsFusion::Metadata meta; // dummy, init() ignores the parameter
  OpsFusion::FusionRuntime rt(flags.xclbin_filename, xclbin_content, "DPU");
  rt.load_state(state_filename_to_use);
  rt.init(meta);
  flags.dbgprint("    ", state_filename_to_use, " loaded");

  if (flags.save_only) {
    return false;
  }

  // Prepare inputs
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<Tensor> in_tensors =
      OpsFusion::MetaUtils::get_input_tensors(rt.get_meta());
  int input_idx = 0;
  for (auto &tensor : in_tensors) {
    ++input_idx;
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);
    std::vector<uint8_t> in(sz, 1);

    // clang-format off

    switch (flags.init_method) {
      default:
      case 0: rand_init_int((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
      case 1: {
        std::string ext = std::string("ddinput-") + std::to_string(input_idx) + std::string(".bin");
        std::string ifname = std::filesystem::path(flags.state_filename).replace_extension(ext).string();
        flags.qprint("Using input file: ", ifname.c_str(), " size ", sz);
        std::ifstream ifs(ifname, std::ios::binary);
        ifs.read((char *)in.data(), sz);
      }
      break;
      case 2: rrand_init_int((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
      case 3: init_int_00s((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
      case 4: init_int_01s((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
      case 5: init_int_10s((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
      case 6: init_int_80s((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
      case 7: init_int_ffs((int8_t *)in.data(), in.size() / sizeof(int8_t)); break;
    }

    // clang-format on

    inputs.push_back(std::move(in));
    tensor.data = inputs.back().data();

    if (flags.write_input) {
      std::string ext = std::string("ddinput-") + std::to_string(input_idx) +
                        std::string(".bin");
      std::string ofname = std::filesystem::path(flags.state_filename)
                               .replace_extension(ext)
                               .string();
      std::ofstream ofs(ofname, std::ios::binary);
      ofs.write((const char *)tensor.data, sz);
      flags.qprint("wrote input to file ", ofname, " size ", sz);
    }
  }

  // Prepare outputs
  std::vector<Tensor> out_tensors =
      OpsFusion::MetaUtils::get_output_tensors(rt.get_meta());
  std::vector<std::vector<uint8_t>> outputs;
  for (auto &tensor : out_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);

    outputs.emplace_back(sz, 1);
    tensor.data = outputs.back().data();
  }

  if (flags.print_summary) {
    std::cout << OpsFusion::MetaUtils::get_summary(rt.get_meta()) << std::endl;
  }

  if (flags.no_execute) {
    if (using_dd_metastate_file) {
      flags.cleanup_other_files(std::string(TMF::string_dd_metastate) +
                                ".state");
    }
    return false;
  }

  if (flags.print_summary || flags.print_perf) {
    std::cout << "Executing for iterations: " << niters << std::endl;
  }

  auto t1_first = std::chrono::steady_clock::now();
  rt.execute(in_tensors, out_tensors);
  auto t2_first = std::chrono::steady_clock::now();

  auto t1_iters = std::chrono::steady_clock::now();
  for (int i = 1; i < niters; ++i) {
    rt.execute(in_tensors, out_tensors);
  }
  auto t2_iters = std::chrono::steady_clock::now();

  if (flags.print_summary || flags.print_perf) {
    auto d_first = t2_first - t1_first;
    auto ms_first = std::chrono::duration<float, std::milli>(d_first).count();

    auto ms_iters = ms_first;
    if (niters > 1) {
      auto d_iters = t2_iters - t1_iters;
      ms_iters = std::chrono::duration<float, std::milli>(d_iters).count() /
                 (niters - 1);
    }
    std::cout << "First Time (ms): " << ms_first << " ";
    std::cout << "Avg. Time (ms): " << ms_iters << std::endl;
  }

  bool compare_failed = false;
  if (flags.write_output || flags.compare_output || flags.gen_golden) {
    int output_idx = 0;
    for (auto &tensor : out_tensors) {
      ++output_idx;
      size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                  size_t{1}, std::multiplies{}) *
                  Utils::get_size_of_type(tensor.dtype);

      if (flags.write_output) {
        std::string ext = std::string("output-") + std::to_string(output_idx) +
                          std::string(".bin");
        std::string ofname = std::filesystem::path(state_filename_to_use)
                                 .replace_extension(ext)
                                 .string();
        std::ofstream ofs(ofname, std::ios::binary);
        ofs.write((const char *)tensor.data, sz);
        flags.qprint("wrote output to file ", ofname, " size ", sz);
      }

      if (flags.gen_golden) {
        std::string ext = std::string("golden-") + std::to_string(output_idx) +
                          std::string(".bin");
        std::string ofname = std::filesystem::path(state_filename_to_use)
                                 .replace_extension(ext)
                                 .string();
        std::ofstream ofs(ofname, std::ios::binary);
        ofs.write((const char *)tensor.data, sz);
        flags.qprint("wrote output to file ", ofname, " size ", sz);
      }

      if (flags.compare_output) {
        std::string ext = std::string("golden-") + std::to_string(output_idx) +
                          std::string(".bin");
        std::string ifname = std::filesystem::path(using_dd_metastate_file
                                                       ? flags.json_filename
                                                       : state_filename_to_use)
                                 .replace_extension(ext)
                                 .string();

        {
          const std::filesystem::path pf(ifname);
          if (!std::filesystem::exists(pf)) {
            ifname = std::filesystem::path(flags.json_filename)
                         .replace_extension(ext)
                         .string();
          }
        }

        {
          const std::filesystem::path pf(ifname);
          if (!std::filesystem::exists(pf)) {
            std::cout << "FAIL"
                      << " "
                      << "does not exist: " << ifname << " " << std::endl;

            compare_failed |= true;
            continue;
          }
        }

        std::ifstream ifs(ifname, std::ios::binary);
        std::vector<char> ibuf;
        ibuf.resize(sz);
        ifs.read(ibuf.data(), sz);
        int rv = memcmp(tensor.data, ibuf.data(), sz);
        std::cout << (rv ? "FAIL" : "PASS") << " "
                  << "compare: " << ifname;
        if (!flags.state_filename.empty()) {
          std::filesystem::path ps(flags.state_filename);
          std::cout << " "
                    << "using " << ps.filename().string();
        }
        std::cout << std::endl;
        compare_failed |= (rv ? true : false);
      }
    }
  }

  if (using_dd_metastate_file) {
    flags.cleanup_other_files(std::string(TMF::string_dd_metastate) + ".state");
  }

  return compare_failed;
}

// test_model
//   test_model_from_ddconfigs
//     test_model_from_ddconfig
//       test_model
static bool test_model_from_ddconfig(const test_model_flags &flags, int niters,
                                     const OpsFusion::Metadata &meta,
                                     const OpsFusion::DDConfig &cfg,
                                     const char *filesubname) {
  std::string filename =
      std::string(TMF::string_dd_metastate) + "-" + filesubname + ".state";

  bool rv = false;
  if (!flags.no_execute) {
    OpsFusion::FusionRuntime rt_cmp;

    std::map<std::string, OpsFusion::SimpleSpan> const_map;
    std::map<std::string, std::vector<char>> const_buffers;
    std::filesystem::path path_to_cache(flags.json_filename);
    OpsFusion::FusionRuntime::build_const_map(
        meta, const_map, const_buffers,
        path_to_cache.replace_filename("").string());
    rt_cmp.compile(meta, "", cfg, const_map);
    rt_cmp.save_state(filename);

    test_model_flags new_flags = flags;
    new_flags.test_configs = false;
    new_flags.state_filename = filename;

    rv = test_model(new_flags, niters);
  }

  flags.cleanup_other_files(filename);

  return rv;
}

// create DDConfigs -> test_model_from_ddconfig
static bool test_model_from_ddconfigs(const test_model_flags &flags, int niters,
                                      OpsFusion::Metadata meta,
                                      std::vector<char> &xclbin_content) {
  int count_failures = 0;
  int count_configs = 0;

  // clang-format off

  // these should match the parsing of "--test_config"
  if (flags.configs_to_test == 0 || (flags.configs_to_test & (1 << 0))) {
      ++count_configs;
      OpsFusion::DDConfig cfg_stress; cfg_stress.xclbin_content = &xclbin_content; cfg_stress.optimize_scratch = true;
      count_failures += test_model_from_ddconfig(flags, niters, meta, cfg_stress, "optimize_scratch")    ?1:0;
  }
  if (flags.configs_to_test == 0 || (flags.configs_to_test & (1 << 1))) {
      ++count_configs;
      OpsFusion::DDConfig cfg_stress; cfg_stress.xclbin_content = &xclbin_content; cfg_stress.eager_mode = true;
      count_failures += test_model_from_ddconfig(flags, niters, meta, cfg_stress, "eager_mode")          ?1:0;
  }
  if (flags.configs_to_test == 0 || (flags.configs_to_test & (1 << 2))) {
      ++count_configs;
      OpsFusion::DDConfig cfg_stress; cfg_stress.xclbin_content = &xclbin_content; cfg_stress.use_lazy_scratch_bo = true;
      count_failures += test_model_from_ddconfig(flags, niters, meta, cfg_stress, "use_lazy_scratch_bo") ?1:0;
  }
  if (flags.configs_to_test == 0 || (flags.configs_to_test & (1 << 3))) {
      ++count_configs;
      OpsFusion::DDConfig cfg_stress; cfg_stress.xclbin_content = &xclbin_content; cfg_stress.en_lazy_constbo = true;
      count_failures += test_model_from_ddconfig(flags, niters, meta, cfg_stress, "en_lazy_constbo")     ?1:0;
  }
  if (flags.configs_to_test == 0 || (flags.configs_to_test & (1 << 4))) {
      ++count_configs;
      OpsFusion::DDConfig cfg_stress; cfg_stress.xclbin_content = &xclbin_content; cfg_stress.dealloc_scratch_bo = true;
      count_failures += test_model_from_ddconfig(flags, niters, meta, cfg_stress, "dealloc_scratch_bo")  ?1:0;
  }
  if (flags.configs_to_test == 0 || (flags.configs_to_test & (1 << 5))) {
      ++count_configs;
      OpsFusion::DDConfig cfg_stress; cfg_stress.xclbin_content = &xclbin_content; cfg_stress.enable_preemption = false;
      count_failures += test_model_from_ddconfig(flags, niters, meta, cfg_stress, "disable_preemption")  ?1:0;
  }

  // clang-format on

  if (count_failures) {
    std::cout << "FAIL with different configs: " << count_failures << " out of "
              << count_configs << std::endl;
  }

  return count_failures ? true : false;
}

static bool make_xclbin_filename(std::string &xclbin_filename,
                                 bool local_xclbin, bool print_debug) {
  const std::filesystem::path pf(xclbin_filename);

  if (local_xclbin) {
    if (std::filesystem::exists(pf)) {
      return true;
    }
    if (print_debug) {
      std::cout << "does not exist: " << pf.string() << std::endl;
    }
  } else {
    if (std::filesystem::exists(pf)) {
      if (print_debug) {
        std::cout << "ignoring: " << pf.string() << std::endl;
      }
    }
  }

  std::string DD_ROOT = Utils::get_env_var("DD_ROOT");
  if (DD_ROOT.empty()) {
    DD_ROOT = ".";
  }

  {
    std::filesystem::path p(DD_ROOT + "/xclbin/stx/release/" +
                            pf.filename().string().c_str());
    if (std::filesystem::exists(p)) {
      xclbin_filename = p.string();
      return true;
    }
    if (print_debug) {
      std::cout << "does not exist: " << p.string() << std::endl;
    }
  }

  {
    std::filesystem::path p(DD_ROOT + "/xclbin/stx/" +
                            pf.filename().string().c_str());
    if (std::filesystem::exists(p)) {
      xclbin_filename = p.string();
      return true;
    }
    if (print_debug) {
      std::cout << "does not exist: " << p.string() << std::endl;
    }
  }

  {
    std::filesystem::path p(DD_ROOT + "/" + pf.filename().string().c_str());
    if (std::filesystem::exists(p)) {
      xclbin_filename = p.string();
      return true;
    }
    if (print_debug) {
      std::cout << "does not exist: " << p.string() << std::endl;
    }
  }

  return false;
}

bool find_state_file(test_model_flags &flags, const std::filesystem::path &p,
                     std::filesystem::file_status s) {
  if (p.extension() != ".state") {
    return false;
  }

  flags.state_filename = p.string();
  return true;
}

bool find_xclbin_file(test_model_flags &flags, const std::filesystem::path &p,
                      std::filesystem::file_status s) {
  if (p.extension() != ".xclbin") {
    return false;
  }

  if (flags.local_xclbin) {
    flags.xclbin_filename = p.string();
  } else {
    flags.xclbin_filename = p.filename().string();
  }

  return true;
}

int main(int argc, const char **argv) {
  if (argc > 1) {
    std::filesystem::path p(argv[1]);

    if (std::filesystem::is_directory(p)) {
      bool a_test_failed = false;
      test_model_flags flags;

      for (auto it{std::filesystem::directory_iterator(argv[1])};
           it != std::filesystem::directory_iterator(); ++it) {

        if (find_state_file(flags, *it, it->status())) {
          for (auto it2{std::filesystem::directory_iterator(argv[1])};
               it2 != std::filesystem::directory_iterator(); ++it2) {

            if (find_xclbin_file(flags, *it2, it2->status())) {
              std::vector<const char *> new_argv;
              new_argv.push_back(argv[0]);
              new_argv.push_back(("--use_state"));
              new_argv.push_back((flags.state_filename.c_str()));
              new_argv.push_back((flags.xclbin_filename.c_str()));
              for (int i = 2; i < argc; ++i) {
                new_argv.push_back(argv[i]);
              }
              int new_argc = static_cast<int>(new_argv.size());
              new_argv.push_back(NULL);

              a_test_failed |= (main(new_argc, new_argv.data()) != 0);
            }
          }
        }
      }
      if (a_test_failed) {
        return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
    }
  }

  // clang-format off

  if (argc < 4) {
    std::filesystem::path pc(argv[0]);
    static const char* indent1 = "  ";
    static const char* indent2 = "      ";

    TMF::print_USAGE(std::cout, pc.filename().string().c_str());

    TMF::print_CSH_ACTIONS(std::cout, indent1, false);

    TMF::print_CSH_OPTIONS(std::cout, indent1, false);

    TMF::print_CSH_DDOPTIONS(std::cout, indent2, false);

    TMF::print_WARNING(std::cout, indent1);

    if (argc > 1) {
      const char* a = argv[1];
      if (*a == '-') ++a;
      if (*a == '-') ++a;
      if (*a == '/') ++a;
      if (*a == '?' || *a == 'h' || *a == 'H') {
        return EXIT_SUCCESS;
      }
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }

  // clang-format on

  bool a_test_failed = false;
  test_model_flags flags;

  try {
    flags.json_filename = std::string(argv[1]);
    flags.state_filename = std::string(argv[2]);
    flags.xclbin_filename = std::string(argv[3]);

    int niters = 1;

    // clang-format off
    for (int i = 4; i < argc; ++i) {
      if (argv[i][0] == '-') {
        std::string a(argv[i]);
        if        (a == TMF::string_load_only      || a == TMF::short_string_load_only     ) { flags.load_only       = true;
        } else if (a == TMF::string_save_only      || a == TMF::short_string_save_only     ) { flags.save_only       = true;
        } else if (a == TMF::string_write_input    || a == TMF::short_string_write_input   ) { flags.write_input     = true;
        } else if (a == TMF::string_write_output   || a == TMF::short_string_write_output  ) { flags.write_output    = true;
        } else if (a == TMF::string_gen_golden     || a == TMF::short_string_gen_golden    ) { flags.gen_golden      = true;
        } else if (a == TMF::string_gen_state      || a == TMF::short_string_gen_state     ) {
          std::cerr << "ERROR: " << argv[i] << " must be the 2nd parameter (instead of 'meta.state')" << std::endl;
          return EXIT_FAILURE;
        } else if (a == TMF::string_use_state      || a == TMF::short_string_use_state     ) {
          std::cerr << "ERROR: " << argv[i] << " must be the 1st parameter (instead of 'meta.json')" << std::endl;
          return EXIT_FAILURE;
        } else if (a == TMF::string_compare_output || a == TMF::short_string_compare_output) { flags.compare_output  = true;
        } else if (a == TMF::string_no_execute     || a == TMF::short_string_no_execute    ) { flags.no_execute      = true;
        } else if (a == TMF::string_no_avoid       || a == TMF::short_string_no_avoid      ) { flags.no_avoid        = true;
        } else if (a == TMF::string_local_xclbin   || a == TMF::short_string_local_xclbin  ) { flags.local_xclbin    = true;
        } else if (a == TMF::string_test_configs   || a == TMF::short_string_test_configs  ) { flags.test_configs    = true;
        } else if (a == TMF::string_test_config    || a == TMF::short_string_test_config   ) { flags.test_configs    = true;
          if (++i >= argc) {
            if (i == argc) { std::cerr << "ERROR: " << argv[i-1] << "needs a #" << std::endl; }
            return EXIT_FAILURE;
          }
          if (argv[i][0] >= '0' && argv[i][0] <= '9')       { flags.configs_to_test |= (1 << (std::atoll(argv[i])-1));
          } else {
            std::string b(argv[i]);
            // these should match the order running of test_model_from_ddconfig
            if        (b == TMF::string_optimize_scratch     || b == TMF::short_string_optimize_scratch    ) { flags.configs_to_test |= (1 << 0) ;
            } else if (b == TMF::string_eager_mode           || b == TMF::short_string_eager_mode          ) { flags.configs_to_test |= (1 << 1) ;
            } else if (b == TMF::string_use_lazy_scratch_bo  || b == TMF::short_string_use_lazy_scratch_bo ) { flags.configs_to_test |= (1 << 2) ;
            } else if (b == TMF::string_en_lazy_constbo      || b == TMF::short_string_en_lazy_constbo     ) { flags.configs_to_test |= (1 << 3) ;
            } else if (b == TMF::string_dealloc_scratch_bo   || b == TMF::short_string_dealloc_scratch_bo  ) { flags.configs_to_test |= (1 << 4) ;
            } else if (b == TMF::string_disable_preemption   || b == TMF::short_string_disable_preemption  ) { flags.configs_to_test |= (1 << 5) ;
            } else {
              TMF::print_CSH_DDOPTIONS(std::cerr, "    ", false);
              std::cerr << "ERROR: unrecognized config: " << b.c_str() << std::endl;
              return EXIT_FAILURE;
            }
          }
        } else if (a == TMF::string_init_method    || a == TMF::short_string_init_method   ) {
          if (++i >= argc) {
            if (i == argc) { std::cerr << "ERROR: " << argv[i-1] << "needs a #" << std::endl; }
            return EXIT_FAILURE;
          }
          if (argv[i][0] >= '0' && argv[i][0] <= '9')                                       { flags.init_method     = static_cast<uint32_t>(atoll(argv[i]));
          }
        } else if (a == TMF::string_print_summary || a == TMF::short_string_print_summary ) { flags.print_summary   = true;
        } else if (a == TMF::string_print_perf    || a == TMF::short_string_print_perf    ) { flags.print_perf      = true;
        } else if (a == TMF::string_print_debug   || a == TMF::short_string_print_debug   ) { flags.print_debug     = true;
        } else if (a == TMF::string_quiet         || a == TMF::short_string_quiet         ) { flags.be_quiet        = true;
        } else if (a == TMF::string_cleanup       || a == TMF::short_string_cleanup       ) { flags.cleanup         = true;
        } else {
          TMF::print_CSH_OPTIONS(std::cerr, "    ", false);
          std::cerr << "ERROR: unrecognized option: " << a.c_str() << std::endl;
          return EXIT_FAILURE;
        }
      } else {
        switch (argv[i][0]) {
        case 'L': flags.load_only           = true; break;
        case 'S': flags.save_only           = true; break;
        case 'I': flags.write_input         = true; break;
        case 'O': flags.write_output        = true; break;
        case 'G': flags.gen_golden          = true; break;
        case 'C': flags.compare_output      = true; break;
        case 'X': flags.no_execute          = true; break;
        case 'A': flags.no_avoid            = true; break;
        case 'M': flags.local_xclbin        = true; break;
        case 'T': flags.test_configs        = true;
          if (argv[i][1] >= '0' && argv[i][1] <= '9') {
            flags.configs_to_test |= (1 << (std::atoll(&argv[i][1])-1));
          }
          break;
        case 'V': flags.print_summary       = true; break;
        case 'P': flags.print_perf          = true; break;
        case 'D': flags.print_debug         = true; break;
        case 'Q': flags.be_quiet            = true; break;
        case 'W': flags.cleanup             = true; break;

        default:
          niters = static_cast<int>(std::atoll(argv[i]));
        }
      }
    }
    // clang-format on

    if (flags.has_action() && flags.no_execute && !flags.cleanup) {
      std::cerr
          << "ERROR: the X flag (no_execute) flag means the G O I C flags "
             "(gen_golden, write output, compare output) are ignored."
          << std::endl;
      return true;
    }

    if (!flags.has_action() && !flags.cleanup) {
      if (!flags.load_only) {
        flags.load_only = !(flags.save_only || flags.state_filename.empty());
      }
      flags.compare_output = flags.print_summary = flags.print_perf = true;
    }

    bool avoid_it = false;

    if (!flags.json_filename.empty()) {
      if (flags.json_filename == TMF::string_use_state ||
          flags.json_filename == TMF::short_string_use_state) {
        flags.json_filename = "";
      } else {
        if (!flags.no_avoid && does_avoid_file_exist(flags.json_filename)) {
          avoid_it = true;
        } else {
          const char *errmsg = verify_is_a_regular_file(flags.json_filename);
          if (errmsg) {
            std::cerr << "ERROR: " << errmsg << ": " << flags.json_filename
                      << std::endl;
            return EXIT_FAILURE;
          }
          flags.qprint("Using json file: ", flags.json_filename);
        }
      }
    }

    if (!flags.state_filename.empty()) {
      if (flags.state_filename == TMF::string_gen_state ||
          flags.state_filename == TMF::short_string_gen_state) {
        flags.state_filename = "";
      } else {
        if (!flags.no_avoid && does_avoid_file_exist(flags.state_filename)) {
          avoid_it = true;
        } else {
          const char *errmsg = verify_is_a_regular_file(flags.state_filename);
          if (errmsg) {
            std::cerr << "ERROR: " << errmsg << ": " << flags.state_filename
                      << std::endl;
            return EXIT_FAILURE;
          }
          flags.qprint("Using state file: ", flags.state_filename);
        }
      }
    }

    flags.dump();

    if (avoid_it) {
      std::cout << "SKIP"
                << " "
                << "because avoid file exists." << std::endl;
      flags.qprint("Finished Successfully by avoidance", "\n");
      return EXIT_SUCCESS;
    }

    {
      if (!make_xclbin_filename(flags.xclbin_filename, flags.local_xclbin,
                                flags.print_debug)) {
        std::cerr << "ERROR: could not locate xclbin file: "
                  << flags.xclbin_filename << std::endl;
        return EXIT_FAILURE;
      }
      flags.qprint("Using xclbin file: ", flags.xclbin_filename);
    }

    a_test_failed = test_model(flags, niters);

  } catch (std::exception &e) {
    std::cout << "ERROR: " << e.what() << std::endl << std::endl;
    return EXIT_FAILURE;
  }
  // clang-format on

  if (a_test_failed) {
    flags.qprint("Finished Successfully with failures", "\n");
    return EXIT_FAILURE;
  }

  flags.qprint("Finished Successfully", "\n");

  return EXIT_SUCCESS;
}
