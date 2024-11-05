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

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "op_const_io.hpp"
#include <op_fuser/fuse_types.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/tfuncs.hpp>
#include <xrt_context/xrt_context.hpp>

#include <xrt/xrt_bo.h>

struct Tensor {
  void *data{nullptr};
  std::vector<size_t> shape;
  std::string dtype;
};

struct OpArgMap {
  enum OpArgType {
    INPUT,
    OUTPUT,
    SCRATCH_PAD,
    CONST_INPUT,
    CONST_KERNEL_PARAM_INPUT,
    CTRL_PKT_BIN,
  };
  OpArgType arg_type;
  size_t xrt_arg_idx;
  size_t onnx_arg_idx;
  size_t offset;
  size_t size; // in bytes
  size_t padding_offset = 0;
};

using save_function = std::function<void(const std::string &, FILE *)>;
using load_function = std::function<FILE *(const std::string &)>;

class OpInterface {
private:
  int num_const = 1;

public:
  OpInterface() {}
  OpInterface(const std::vector<std::string> in_dtypes,
              const std::vector<std::string> out_dtypes){};
  virtual ~OpInterface() = default;

  /// @brief Initialize the internal input buffers of each node. This is called
  /// once during setup initialization.
  virtual void
  initialize_inputs(const std::vector<Tensor> &inputs,
                    const std::map<std::string, std::any> &attr = {}) {}

  /// @brief Format and copy the NPU Output to user tensor after the execution.
  /// Default action is direct copy of data to user tensor.
  /// @param out_tensor User Tensor
  /// @param hw_out_ptr Pointer to the internal tensor produced by NPU
  /// @param sz         Size of the internel tensor produced by NPU
  /// @param tensor_idx Output index of the tensor. 0 if node has a single
  /// output
  /// @param attr       Node Attributes
  virtual void format_output(const Tensor &out_tensor, void *hw_out_ptr,
                             size_t sz, size_t tensor_idx,
                             const std::map<std::string, std::any> &attr = {});
  virtual void
  initialize_const_params(ConstBufferIO &io,
                          const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {}) = 0;

  virtual void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {}) = 0;

  virtual const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const = 0;

  virtual const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const = 0;

  virtual std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) const = 0;

  virtual std::vector<uint8_t>
  get_ctrl_pkts(std::vector<Tensor> &input, std::vector<Tensor> &output,
                const std::map<std::string, std::any> &attr = {}) const {
    return {};
  };

  virtual std::vector<CtrlPktPatchInfo> get_ctrl_pkt_patch_info(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const {
    return {};
  };

  virtual const std::map<std::string, std::any> &get_attr() const {
    static const std::map<std::string, std::any> empty_map;
    return empty_map;
  }

  virtual void execute(std::vector<Tensor> &input,
                       std::vector<Tensor> &output) {}

  virtual void execute(std::vector<xrt::bo> &input,
                       std::vector<xrt::bo> &output) {}

  static void set_dd_base_dir(const std::string &dir);

  static std::string get_dd_base_dir();
  int GetNumConst() { return num_const; }

protected:
  std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> xrt_ctx_;
  static ryzenai::dynamic_dispatch::instruction_registry instr_reg_;
  static std::string dd_base_dir;
  void SetNumConst(int value) { num_const = value; };
};

std::string convert_argtype_to_string(OpArgMap::OpArgType arg_type);
std::string cvt_to_string(const OpArgMap &arg);
std::string cvt_to_string(const std::vector<OpArgMap> &argmap);

// Utility to invoke OpInterface methods with verbose error checks.
template <typename Func, typename... Args>
static auto dd_invoke_op_method(const std::string &func_name,
                                const char *srcfile, size_t line_no,
                                const OpsFusion::Metadata::OpInfo &op_info,
                                Func &&func, OpInterface *op, Args &&...args) {
  OpsFusion::LifeTracer lt(
      OpsFusion::dd_format("Invoking {}() for op:{}, op_type:{}", func_name,
                           op_info.name, op_info.type));

  try {
    return func(op, std::forward<Args>(args)...);
  } catch (std::exception &e) {
    throw std::runtime_error(OpsFusion::dd_format(
        "[{}:{}] Invoking {}() failed !!\nDetails:\n  Op "
        "Name: {}\n  Op Type: {}\n  Error: {}",
        srcfile, line_no, func_name, op_info.name, op_info.type, e.what()));
  } catch (...) {
    throw std::runtime_error(OpsFusion::dd_format(
        "[{}:{}] Invoking {}() failed !!\nDetails:\n  Op "
        "Name: {}\n  Op Type: {}\n  Error: Unknown Exception",
        srcfile, line_no, func_name, op_info.name, op_info.type));
  }
}

// Invoke OpInterface method with verbose error check
#define DD_INVOKE_OPMETHOD(method_name, op_object, op_info, ...)               \
  dd_invoke_op_method("OpInterface::" #method_name, __FILE__, __LINE__,        \
                      op_info, std::mem_fn(&OpInterface::method_name),         \
                      op_object, __VA_ARGS__)

// Invoke OpInterface method with verbose error check
#define DD_INVOKE_OVERLOADED_OPMETHOD(method_name, signature, op_object,       \
                                      op_info, ...)                            \
  dd_invoke_op_method("OpInterface::" #method_name, __FILE__, __LINE__,        \
                      op_info,                                                 \
                      std::mem_fn<signature>(&OpInterface::method_name),       \
                      op_object, __VA_ARGS__)
