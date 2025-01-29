// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
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
#include <memory>

#include "op_interface.hpp"
#include "op_policy.hpp"
#include "unordered_map"
#include <op_fuser/fuse_types.hpp>
#include <stdexcept>

using namespace std::literals::string_literals;

namespace OpsFusion {

class OpBuilder {
public:
  OpBuilder() = default;
  virtual ~OpBuilder() = default;

  static void init();

  // TODO : What is the right info to be passed to the builder ?
  static std::unique_ptr<OpInterface>
  create(const std::string &op_name, const Metadata::OpInfo &op_info,
         const std::map<std::string, Metadata::OffsetInfo> &tensor_map);

  static std::unique_ptr<OpInterface> create(const OpParams &);

  static bool is_supported(const std::string &op_type,
                           const std::vector<std::string> &types,
                           const std::map<std::string, std::any> &attr);

  template <typename T, typename... Args>
  static void register_op(std::string op_name,
                          const std::vector<std::string> &type_info,
                          const std::vector<int> &pos, Args &&...args);

private:
  bool static is_val_skipped(std::vector<int> pos);
  using UPTRIOpContainer = std::vector<std::unique_ptr<ICreatePolicy>>;
  inline static std::unordered_map<std::string, UPTRIOpContainer> op_registry =
      {};
};

template <class T, size_t N>
void static reg_op_with_def_policy(std::string op_name,
                                   const std::vector<std::string> &type_info,
                                   const std::vector<int> &pos) {

  using Type = DefaultCreate<T, N>;
  OpBuilder::register_op<Type>(op_name, type_info, pos);
}

template <class T, size_t N>
void static reg_op_with_def_policy_wo_attr(
    std::string op_name, const std::vector<std::string> &type_info,
    const std::vector<int> &pos) {

  using Type = DefaultCreateWOAttr<T, N>;
  OpBuilder::register_op<Type>(op_name, type_info, pos);
}

void static reg_op_with_custom_policy(std::string op_name,
                                      const std::vector<std::string> &type_info,
                                      const std::vector<int> &pos,
                                      const OpFuncTy &func) {
  using Type = CustomCreate;
  OpBuilder::register_op<Type>(op_name, type_info, pos, func);
}

#define REGISTER_KNOWN_OPS(Func)                                               \
  static int tmp = []() {                                                      \
    Func();                                                                    \
    return 0;                                                                  \
  }();

} // namespace OpsFusion
