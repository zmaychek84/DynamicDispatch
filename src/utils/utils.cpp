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
#include <chrono>
#include <string>
#include <unordered_map>

#include <utils/utils.hpp>

namespace Utils {
std::string get_env_var(const std::string &var,
                        const std::string &default_val) {
#ifdef _WIN32
  char *value = nullptr;
  size_t size = 0;
  errno_t err = _dupenv_s(&value, &size, var.c_str());
  std::string result =
      (!err && (value != nullptr)) ? std::string{value} : default_val;
  free(value);
#else
  const char *value = std::getenv(var.c_str());
  std::string result = (value != nullptr) ? std::string{value} : default_val;
#endif
  return result;
}

int64_t ceil_for_me(int64_t x, int64_t y) {
  return int64_t(y * std::ceil(x * 1.0 / y));
}

// Align the 'n' to a multiple of 'A'
// new_n >= n
// new_n % A = 0
size_t align_to_next(size_t n, size_t alignment) {
  return ((n + alignment - 1) / alignment) * alignment;
}

size_t get_size_of_type(const std::string &type) {
  static const std::unordered_map<std::string, size_t> elem_size{
      {"int8", 1},     {"uint8", 1},   {"int16", 2}, {"uint16", 2},
      {"int32", 4},    {"uint32", 4},  {"int64", 8}, {"uint64", 8},
      {"bfloat16", 2}, {"float32", 4}, {"float", 4}, {"double", 8},
      {"float64", 8}};
  if (elem_size.find(type) == elem_size.end()) {
    throw std::runtime_error("get_size_of_type - Invalid type : " + type);
  }
  auto sz = elem_size.at(type);
  return sz;
}

std::vector<std::string> split_string(const std::string &msg,
                                      const std::string &delim) {
  size_t start = 0;
  size_t end = std::string::npos;
  std::vector<std::string> tokens;
  while (true) {
    end = msg.find(delim, start);
    if (end == std::string::npos) {
      auto sub = msg.substr(start, end - start);
      tokens.push_back(sub);
      break;
    }
    auto sub = msg.substr(start, end - start);
    tokens.push_back(sub);
    start = end + delim.size();
  }
  return tokens;
}

std::string remove_whitespaces(std::string x) {
  auto iter = std::remove(x.begin(), x.end(), ' ');
  x.erase(iter, x.end());
  return x;
}

void dumpBinary(void *src, size_t length, std::string &filePath) {
  std::ofstream ofs(filePath, std::ios::binary);
  size_t chunk_size = 1024;
  char *ptr = (char *)src;
  for (int i = 0; i < length / chunk_size; ++i) {
    ofs.write((char *)src, 1024);
    ptr += chunk_size;
  }
  ofs.write(ptr, length % chunk_size);
}

std::string generateCurrTimeStamp() {
  auto now = std::chrono::high_resolution_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch())
                       .count();
  return std::to_string(timestamp);
}

} // namespace Utils
