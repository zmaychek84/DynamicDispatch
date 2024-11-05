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
#include <filesystem>
#include <utils/file_ptr.hpp>
namespace Utils {
FILE *create_tmpfile();
void dump_to_tmpfile(FILE *file, char *data, size_t size);
void save_tmpfile_on_disk(const std::filesystem::path &path, FILE *file);

template <typename T> struct binary_io {
  using char_type = T;
  static std::vector<char_type> slurp_binary(FILE *file) {
    fseek64(file, 0, SEEK_SET);
    fseek64(file, 0, SEEK_END);
    auto size = ftell64(file);
    fseek64(file, 0, SEEK_SET);
    auto buffer = std::vector<char_type>((size_t)size / sizeof(char_type));
    if (size != 0) {
      fread(buffer.data(), 1, size, file);
    }
    return buffer;
  }
};

}; // namespace Utils
