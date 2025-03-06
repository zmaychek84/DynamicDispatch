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

#pragma once

#include "detail/tfunc_impl.hpp"
#include <fstream>
#include <map>
#include <utils/logging.hpp>
#include <vector>

namespace OpsFusion {

template <typename... Args>
static std::string dd_format(const std::string &msg, Args &&...args) {
  return detail::dd_format_impl(msg, std::forward<Args>(args)...);
}

// A RAII-Wrapper to mark begin & end of life of anything.
struct LifeTracer {
  LifeTracer(std::string msg) : msg_(std::move(msg)) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format("{} ... START", msg_));
  }
  ~LifeTracer() { RYZENAI_LOG_TRACE(OpsFusion::dd_format("{} ... END", msg_)); }

private:
  std::string msg_;
};

// Read binary file to a vector
template <typename T = char>
std::vector<T> read_bin_file(const std::string &filename) {

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Opening file: {} ...", filename));
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error(
        OpsFusion::dd_format("Couldn't open file for reading : {}", filename));
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Loading data from {}...", filename));
  std::vector<T> dst;

  try {
    ifs.seekg(0, ifs.end);
    auto size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    dst.resize(size / sizeof(T));
    ifs.read((char *)dst.data(), size);
  } catch (std::exception &e) {
    throw std::runtime_error(OpsFusion::dd_format(
        "Failed to read contents from file {}, error: {}", filename, e.what()));
  }
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Loading data from {} ... DONE", filename));

  return dst;
}

template <typename T = char>
std::vector<T> read_bin_file_from_big_file(const std::string &dir_path,
                                           const std::string &filename_only,
                                           size_t offset, size_t size) {
  if (static_cast<long>(offset) < 0) {
    throw std::runtime_error(OpsFusion::dd_format(
        "read_bin_file_from_big_file(): offset is too large for fseek",
        offset));
  }

  static std::vector<std::pair<std::string, FILE *>> v_filename_to_fh;

  FILE *fh = NULL;
  static FILE *last_fh = NULL;

  std::string filename;
  if (dir_path.empty()) {
    filename = filename_only;
  } else {
    filename = dir_path + "/" + filename_only;
  }

  for (int i = 0; i < v_filename_to_fh.size(); ++i) {
    if (v_filename_to_fh[i].first == filename) {
      fh = v_filename_to_fh[i].second;
      break;
    }
  }

  if (fh == NULL) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Opening big file: {} ...", filename));
    fh = fopen(filename.c_str(), "rb");
    if (fh == NULL) {
      throw std::runtime_error(OpsFusion::dd_format(
          "Couldn't open big file for reading : {}", filename));
    }
    v_filename_to_fh.push_back(std::pair<std::string, FILE *>(filename, fh));
    last_fh = fh;
  }

  if (fh != last_fh) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format("Using big file: {} ...", filename));
    last_fh = fh;
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Loading data from {}...", filename));
  std::vector<T> dst;

  try {
    dst.resize(size / sizeof(T));

    fseek(fh, static_cast<long>(offset), SEEK_SET);
    fread((char *)dst.data(), sizeof(T), size / sizeof(T), fh);

  } catch (std::exception &e) {
    throw std::runtime_error(OpsFusion::dd_format(
        "Failed to read contents from file {}, error: {}", filename, e.what()));
  }
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Loading data from {} ... DONE", filename));

  return dst;
}

template <typename srcT, typename Func>
auto for_each(const std::vector<srcT> &src, Func &&f) {
  using dstT = decltype(f(srcT{}));
  std::vector<dstT> res;
  res.reserve(src.size());
  for (const auto &item : src) {
    res.push_back(f(item));
  }
  return res;
}

template <typename Func, typename... Args>
static auto dd_invoke_impl(const std::string &func_name, const char *srcfile,
                           size_t line_no, Func &&f, Args &&...args) {
  LifeTracer lt(dd_format("Invoking {}", func_name));

  try {
    return f(std::forward<Args>(args)...);
  } catch (std::exception &e) {
    throw std::runtime_error(
        dd_format("[{}:{}] Invoking {}() failed with error: {}", srcfile,
                  line_no, func_name, e.what()));
  } catch (...) {
    throw std::runtime_error(
        dd_format("[{}:{}] Invoking {}() failed with Unknown Exception",
                  srcfile, line_no, func_name));
  }
}

} // namespace OpsFusion

// Following helper macros can be used to access elements from containers
// If access throws exception, it prints more details with the exception for
// debugging

// Equivalent to .at() method of std::vector/std::array
#define ARRAY_AT(x, idx)                                                       \
  OpsFusion::detail::vector_get_value_at(x, idx, #x, __FILE__, __LINE__)

// Equivalent to index access of a new/malloc buffer
#define PTR_AT(ptr, sz, idx)                                                   \
  OpsFusion::detail::ptr_get_at(ptr, sz, idx, #ptr, __FILE__, __LINE__)

// Equivalent to .at() method of std::map/std::unordered_map
#define MAP_AT(x, key)                                                         \
  OpsFusion::detail::map_get_value_at(x, key, #x, __FILE__, __LINE__)

// Throw with source location
#define DD_THROW(msg) OpsFusion::detail::throw_loc(msg, __FILE__, __LINE__)

#define DD_WARNING(msg)                                                        \
  dd_format("[WARNING]: {} @{}:{}\n", msg, __FILE__, __LINE__)
#define DD_INFO(msg) dd_format("[INFO]: {}", msg)

// Throw is condition fails
#define DD_ASSERT(cond, msg)                                                   \
  if (!(cond)) {                                                               \
    DD_THROW(msg);                                                             \
  }

// Throw is condition fails
#define DD_THROW_IF(cond, msg)                                                 \
  if ((cond)) {                                                                \
    DD_THROW(msg);                                                             \
  }

// Invoke an external function with exception check
// Eg : auto res = DD_INVOKE(add_func, 2, 3);
#define DD_INVOKE(func, ...) dd_invoke_impl(#func, &func, __VA_ARGS__)

// Invoke an external class member func with exception check
// Eg : auto res = DD_INVOKE_MEMFN(classA::methodB, objA, args1, args2)
#define DD_INVOKE_MEMFN(func, ...)                                             \
  dd_invoke_impl(#func, std::mem_fn(&func), __VA_ARGS__)
