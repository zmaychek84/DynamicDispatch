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

#pragma once

#include <array>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename V>
static std::ostream &operator<<(std::ostream &os, const std::vector<V> &vec) {
  for (const auto &v : vec) {
    os << v << ", ";
  }
  return os;
}

template <typename K, typename V>
static std::ostream &operator<<(std::ostream &os, const std::map<K, V> &dict) {
  for (const auto &[k, v] : dict) {
    os << k << " : " << v << "\n";
  }
  return os;
}

template <typename K, typename V>
static std::ostream &operator<<(std::ostream &os,
                                const std::map<K, std::vector<V>> &dict) {
  for (const auto &[k, v] : dict) {
    os << k << " : " << v << "\n";
  }
  return os;
}

namespace OpsFusion {
namespace detail {

static std::string combine_file_line(const std::string &file, size_t line) {
  return "[" + file + ":" + std::to_string(line) + "]";
}

template <typename T>
static T &ptr_get_at(T *arr, size_t sz, size_t idx, const char *name,
                     const char *file, size_t line) {

  if (idx >= sz) {
    std::ostringstream oss;
    oss << file << ":" << line << " [ERROR] array out-of-bound access"
        << "\n"
        << "Details - name: " << name << ", size: " << sz << ", idx: " << idx
        << std::endl;
    throw std::runtime_error(oss.str());
  }
  return arr[idx];
}

template <typename T>
static T &vector_get_value_at(std::vector<T> &vec, size_t idx, const char *name,
                              const char *file, size_t line) {
  return ptr_get_at(vec.data(), vec.size(), idx, name, file, line);
}

template <typename T>
static const T &vector_get_value_at(const std::vector<T> &vec, size_t idx,
                                    const char *name, const char *file,
                                    size_t line) {
  return ptr_get_at(vec.data(), vec.size(), idx, name, file, line);
}

template <typename T, size_t N>
static T &vector_get_value_at(std::array<T, N> &vec, size_t idx,
                              const char *name, const char *file, size_t line) {
  return ptr_get_at(vec.data(), N, idx, name, file, line);
}

template <typename T, size_t N>
static const T &vector_get_value_at(const std::array<T, N> &vec, size_t idx,
                                    const char *name, const char *file,
                                    size_t line) {
  return ptr_get_at(vec.data(), N, idx, name, file, line);
}

template <typename T, size_t N>
static T &vector_get_value_at(T (&vec)[N], size_t idx, const char *name,
                              const char *file, size_t line) {
  return ptr_get_at(&(vec[0]), N, idx, name, file, line);
}

template <typename T, size_t N>
static const T &vector_get_value_at(const T (&vec)[N], size_t idx,
                                    const char *name, const char *file,
                                    size_t line) {
  return ptr_get_at(&(vec[0]), N, idx, name, file, line);
}

template <typename T>
static T &c_array_get_value_at(T vec[], size_t N, size_t idx, const char *name,
                               const char *file, size_t line) {
  return ptr_get_at(&vec[0], N, idx, name, file, line);
}

template <typename T>
static const T &c_array_get_value_at(const T vec[], size_t N, size_t idx,
                                     const char *name, const char *file,
                                     size_t line) {
  return ptr_get_at(&vec[0], N, idx, name, file, line);
}

template <typename Container, typename K = typename Container::key_type,
          typename V = typename Container::mapped_type>
static V &map_get_value_at(Container &container, const K &key, const char *name,
                           const char *file, size_t line) {
  auto iter = container.find(key);
  if (iter == container.end()) {
    std::ostringstream oss;
    oss << combine_file_line(file, line) << " [ERROR] Invalid Key Access "
        << "(Container: " << name << ", Key: " << key
        << ", Size: " << container.size() << ")\n";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

template <typename Container, typename K = typename Container::key_type,
          typename V = typename Container::mapped_type>
static const V &map_get_value_at(const Container &container, const K &key,
                                 const char *name, const char *file,
                                 size_t line) {
  auto iter = container.find(key);
  if (iter == container.end()) {
    std::ostringstream oss;
    oss << combine_file_line(file, line) << " [ERROR] Invalid Key Access "
        << "(Name: " << name << ", Key: " << key
        << ", Size: " << container.size() << ")\n";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

static std::string cvt_to_string(const std::string &str) { return str; }
static std::string cvt_to_string(const char *str) { return str; }
template <typename T> static std::string cvt_to_string(T num) {
  std::ostringstream oss;
  oss << num;
  return oss.str();
}
template <typename T> static std::string cvt_to_string(std::vector<T> v) {
  std::ostringstream oss;
  oss << "[ ";
  for (const auto &i : v) {
    oss << cvt_to_string(i) << ", ";
  }
  oss << " ]";
  return oss.str();
}
template <typename T, typename U>
static std::string cvt_to_string(std::map<T, U> m) {
  std::ostringstream oss;
  for (const auto &[k, v] : m) {
    oss << cvt_to_string(k) << " : " << cvt_to_string(v) << "\n";
  }
  return oss.str();
}

template <typename... Args>
static std::string dd_format_impl(const std::string &msg, Args &&...args) {
  constexpr size_t sz = sizeof...(args);
  if constexpr (sz == 0) {
    return msg;
  } else {
    static_assert(sz > 0, "Size > 0");
    std::string sargs[] = {cvt_to_string(args)...};

    size_t start = 0;
    auto end = std::string::npos;
    std::vector<std::string> tokens;
    while (true) {
      end = msg.find("{}", start);
      if (end == std::string::npos) {
        auto sub = msg.substr(start, end - start);
        tokens.push_back(sub);
        break;
      }
      auto sub = msg.substr(start, end - start);
      tokens.push_back(sub);
      tokens.push_back("{}");
      start = end + 2;
    }

    std::string res;
    size_t arg_idx = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i] == "{}") {
        auto sub = arg_idx < sz ? sargs[arg_idx] : tokens[i];
        res += sub;
        arg_idx++;
      } else {
        res += tokens[i];
      }
    }
    return res;
  }
}

static void throw_loc(const std::string &msg, const char *srcfile,
                      size_t line_no) {
  throw std::runtime_error(
      dd_format_impl("[{}:{}] [ERROR] {}", srcfile, line_no, msg));
}

} // namespace detail
} // namespace OpsFusion
