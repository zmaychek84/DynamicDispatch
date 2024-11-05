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
#include <iomanip>

#include <ops/op_interface.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

ryzenai::dynamic_dispatch::instruction_registry OpInterface::instr_reg_;
std::string OpInterface::dd_base_dir{};

void OpInterface::format_output(const Tensor &out_tensor, void *hw_out_ptr,
                                size_t sz, size_t tensor_idx,
                                const std::map<std::string, std::any> &attr) {
  auto out_tensor_sz =
      std::accumulate(out_tensor.shape.begin(), out_tensor.shape.end(),
                      size_t{1}, std::multiplies{}) *
      Utils::get_size_of_type(out_tensor.dtype);
  DD_ASSERT(
      out_tensor_sz <= sz,
      OpsFusion::dd_format("Size mismatch in format_output() default impl : "
                           "user_tensor_size:{} v/s dd_tensor_size:{}",
                           out_tensor_sz, sz));
  memcpy(out_tensor.data, hw_out_ptr, out_tensor_sz);
}

void OpInterface::set_dd_base_dir(const std::string &dir) {
  //   DD_THROW_IF(!dir.empty() && !std::filesystem::exists(dir),
  //                OpsFusion::dd_format("Dir {} doesn't exist",
  //                std::quoted(dir)));

  if (dd_base_dir.empty()) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Setting DoD base dir to {}", std::quoted(dir)));
  } else {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("[WARNING] Overwriting DoD base dir from {} to {}",
                             std::quoted(dd_base_dir), std::quoted(dir)));
  }

  dd_base_dir = dir;
}

std::string OpInterface::get_dd_base_dir() {
  bool is_dd_base_dir_set = !dd_base_dir.empty();
  if (is_dd_base_dir_set) {
    return dd_base_dir;
  }

  RYZENAI_LOG_TRACE(
      "DoD base dir is not set. Checking the DD_ROOT env variable.");
  std::string dd_root_env = Utils::get_env_var("DD_ROOT");
  if (dd_root_env.empty()) {
    DD_THROW("DoD base dir is not set. Use OpInterface::set_dd_base_dir(dir) "
             "API or set DD_ROOT env variable.");
  }

  return dd_root_env;
}

std::string convert_argtype_to_string(OpArgMap::OpArgType arg_type) {

  std::string arg;
  switch (arg_type) {
  case OpArgMap::OpArgType::INPUT:
    arg = "in";
    break;
  case OpArgMap::OpArgType::OUTPUT:
    arg = "out";
    break;
  case OpArgMap::OpArgType::SCRATCH_PAD:
    arg = "scratch";
    break;
  case OpArgMap::OpArgType::CONST_INPUT:
    arg = "const";
    break;
  case OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT:
    arg = "super_instr";
    break;
  case OpArgMap::OpArgType::CTRL_PKT_BIN:
    arg = "ctrl_pkt";
    break;
  default:
    DD_THROW("Invalide arg_type conversion to string");
    break;
  }

  return arg;
}

std::string cvt_to_string(const OpArgMap &arg) {
  return OpsFusion::dd_format(
      "argtype:{}, xrt_id:{}, onnx_id:{}, offset:{}, size:{}",
      convert_argtype_to_string(arg.arg_type), arg.xrt_arg_idx,
      arg.onnx_arg_idx, arg.offset, arg.size);
}

std::string cvt_to_string(const std::vector<OpArgMap> &argmap) {
  std::ostringstream oss;
  size_t idx = 0;
  for (const auto &arg : argmap) {
    oss << OpsFusion::dd_format("{} - {}", idx, cvt_to_string(arg))
        << std::endl;
    idx++;
  }
  return oss.str();
}
