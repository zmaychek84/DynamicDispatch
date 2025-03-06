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

#include "passes.hpp"

#include <op_fuser/fuse_types.hpp>

namespace OpsFusion {

void optimize_scratch_buffer(Metadata &meta) {
  const std::string DD_BUFFER_REUSE_DEFAULT_VERSION = "v3";

  std::string buffer_reuse_version = Utils::get_env_var(
      "DD_BUFFER_REUSE_VERSION", DD_BUFFER_REUSE_DEFAULT_VERSION);

  if (buffer_reuse_version == DD_BUFFER_REUSE_DEFAULT_VERSION) {
    optimize_scratch_buffer_contiguous(meta);
  } else {
    optimize_scratch_buffer_bucket(meta, buffer_reuse_version);
  }
}

} // namespace OpsFusion
