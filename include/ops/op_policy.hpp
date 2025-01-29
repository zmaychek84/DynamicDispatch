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
#include "op_interface.hpp"
#include <any>
#include <assert.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace OpsFusion {

using namespace std::literals::string_literals;

// Unpack caller for variadic arguments
template <size_t num_args> struct unpack_caller {

  template <typename FuncType, typename T, size_t... I, typename... Args>
  decltype(auto) call(FuncType &f, const std::vector<T> &args,
                      std::index_sequence<I...>, Args... add) {
    return f(args[I]..., std::forward<Args>(add)...);
  }

public:
  template <typename FuncType, typename T, typename... Args>
  decltype(auto) operator()(FuncType &f, const std::vector<T> &args,
                            Args &&...add) {
    assert(args.size() == num_args &&
           "Args count in template argument must match with type args size.");
    return call(f, args, std::make_index_sequence<num_args>{},
                decltype(add)(add)...);
  }
};

// Unique pointer wrapper for creating unique_ptr instances
template <typename T> struct UniquePtrWrapper {
  UniquePtrWrapper() {}

  template <typename... Args> std::unique_ptr<T> operator()(Args... args) {
    return std::make_unique<T>(decltype(args)(args)...);
  }
};

/// @brief Struct to hold operation parameters
struct OpParams {

  std::string op_name;
  Metadata::OpInfo op_info;
  std::map<std::string, Metadata::OffsetInfo> tensor_map;

  OpParams(const std::string &op_name, const Metadata::OpInfo &op_info,
           const std::map<std::string, Metadata::OffsetInfo> &tensor_map)
      : op_name(op_name), op_info(op_info), tensor_map(tensor_map)

  {}
};
// / @brief Extract datatypes of all arguments of the op from Nodes' attributes.
// / This is valid only for DD-vaip-cpp flow.
static std::vector<std::string>
extract_dtypes_from_attrs(const OpsFusion::Metadata::OpInfo &op_info) {
  const auto &in_arg_dtypes = std::any_cast<const std::vector<std::string> &>(
      MAP_AT(op_info.attr, "in_dtypes"));
  const auto &out_arg_dtypes = std::any_cast<const std::vector<std::string> &>(
      MAP_AT(op_info.attr, "out_dtypes"));

  std::vector<std::string> arg_dtypes;
  arg_dtypes.insert(arg_dtypes.end(), in_arg_dtypes.begin(),
                    in_arg_dtypes.end());
  arg_dtypes.insert(arg_dtypes.end(), out_arg_dtypes.begin(),
                    out_arg_dtypes.end());

  return arg_dtypes;
}

/// @brief Extract datatypes of all arguments of the op from Nodes' tensors.
/// This is valid only for DD-vaip-Python flow.
static std::vector<std::string> extract_dtypes_from_tensors(
    const OpsFusion::Metadata::OpInfo &op_info,
    const std::map<std::string, Metadata::OffsetInfo> &tensor_map) {
  std::vector<std::string> arg_dtypes;
  auto args = OpsFusion::get_op_args(op_info);
  for (const auto &arg : args) {
    arg_dtypes.push_back(MAP_AT(tensor_map, arg).dtype);
  }
  return arg_dtypes;
}

/// @brief Wrapper to extract datatypes of all arguments of the op
static std::vector<std::string> extract_arg_dtypes(
    const OpsFusion::Metadata::OpInfo &op_info,
    const std::map<std::string, Metadata::OffsetInfo> &tensor_map) {
  std::vector<std::string> arg_dtypes;
  if (op_info.attr.find("in_dtypes") != op_info.attr.end() &&
      op_info.attr.find("out_dtypes") != op_info.attr.end()) {
    arg_dtypes = extract_dtypes_from_attrs(op_info);
  } else {
    arg_dtypes = extract_dtypes_from_tensors(op_info, tensor_map);
  }
  return arg_dtypes;
}

using OpFuncTy = std::function<std::unique_ptr<OpInterface>(const OpParams &)>;

/// @brief Interface for the creation policy class
class ICreatePolicy {
public:
  ICreatePolicy(const std::vector<std::string> &op_types,
                const std::vector<int> &pos, const std::string &op_name)
      : m_types(op_types), m_pos(pos), m_op_name(op_name) {

    if (!(m_pos.size() == 0 || m_pos.size() == m_types.size())) {
      throw std::runtime_error("size mismatch between types and position for " +
                               op_name);
    }
  }
  // Dry run function
  // virtual void dry_run(const OpParams &op_params) {
  /*TODO IMPL*/
  //};

  /// @brief Pure virtual function for creating an instance of OpInterface
  virtual std::unique_ptr<OpInterface> create(const OpParams &op_params) = 0;

  /// @brief Verify function checks for matching types in op_types
  virtual bool verify(const std::vector<std::string> &op_types) {
    for (int i = 0; i < m_pos.size(); i++) {
      auto pos = m_pos[i];
      if (pos == -1 || ARRAY_AT(op_types, pos) == m_types[i]) {
        continue;
      }
      return false;
    }
    return true;
  }

  std::vector<int> get_pos() { return m_pos; }
  virtual ~ICreatePolicy() {}

protected:
  std::vector<std::string> m_types;
  std::vector<int> m_pos;
  std::string m_op_name;
};

/// @brief Default creation policy with attributes
template <typename OpTy, size_t N> class DefaultCreate : public ICreatePolicy {
public:
  DefaultCreate(const std::vector<std::string> &op_types,
                const std::vector<int> &pos, const std::string &op_name)
      : ICreatePolicy(op_types, pos, op_name) {}

  /// @brief Override create function from ICreatePolicy
  virtual std::unique_ptr<OpInterface> create(const OpParams &op_params) {
    auto op_types = extract_arg_dtypes(op_params.op_info, op_params.tensor_map);

    if (!verify(op_types)) {
      return nullptr;
    }

    auto obj = UniquePtrWrapper<OpTy>();
    return m_caller(obj, m_types, false, op_params.op_info.attr);
  }
  virtual ~DefaultCreate() {}

private:
  unpack_caller<N> m_caller;
};

/// @brief Custom creation policy
class CustomCreate : public ICreatePolicy {
public:
  CustomCreate(const std::vector<std::string> &op_types,
               const std::vector<int> &pos, const std::string &op_name,
               const OpFuncTy &func)
      : ICreatePolicy(op_types, pos, op_name), m_func(func) {}

  virtual std::unique_ptr<OpInterface> create(const OpParams &op_params) {

    auto op_types = extract_arg_dtypes(op_params.op_info, op_params.tensor_map);

    if (!verify(op_types)) {
      return nullptr;
    }
    return m_func(op_params);
  }
  virtual ~CustomCreate() {}

private:
  OpFuncTy m_func;
};

/// @brief Default creation policy with attributes.
template <typename OpTy, size_t N>
class DefaultCreateWOAttr : public ICreatePolicy {

public:
  DefaultCreateWOAttr(const std::vector<std::string> &op_types,
                      const std::vector<int> &pos, const std::string &op_name)
      : ICreatePolicy(op_types, pos, op_name) {}

  /// @brief Override create function from ICreatePolicy.
  virtual std::unique_ptr<OpInterface> create(const OpParams &op_params) {
    auto op_types = extract_arg_dtypes(op_params.op_info, op_params.tensor_map);

    if (!verify(op_types)) {
      return nullptr;
    }

    auto obj = UniquePtrWrapper<OpTy>();
    return m_caller(obj, m_types, false);
  }
  virtual ~DefaultCreateWOAttr() {}

private:
  unpack_caller<N> m_caller;
};
} // namespace OpsFusion
