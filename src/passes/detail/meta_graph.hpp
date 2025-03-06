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

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>

#include <ops/op_interface.hpp>
#include <unordered_set>
#include <utils/tfuncs.hpp>

namespace OpsFusion {
namespace Pass {
namespace detail {

/*
  Graph API based on Metadata Structure.
  This class will act as a base API to treat Metadata as a graph.
  This will be useful if some passes require to traverse through the metadata
  like a graph.
*/

class MetaGraph {
public:
  MetaGraph() = default;
  MetaGraph(OpsFusion::Metadata meta) : meta_(std::move(meta)) {
    fill_node_inputs_outputs();
    fill_producers_consumers();
    fill_op_index();
  }

  /// @brief Get the input tensor names of the graph
  /// @return
  const std::vector<std::string> &get_input_tensors() const {
    return MAP_AT(meta_.fused_tensors, "in").packed_tensors;
  }

  /// @brief Get the output tensor names of the graph
  /// @return
  const std::vector<std::string> &get_output_tensors() const {
    return MAP_AT(meta_.fused_tensors, "out").packed_tensors;
  }

  /// @brief Get the input tensor names of an Op in the graph
  /// @param op_name Name of the Op
  /// @return
  const std::vector<std::string> &
  get_op_inputs(const std::string &op_name) const {
    return MAP_AT(node_inputs_, op_name);
  }

  /// @brief Get the input tensor names of an Op in the graph
  /// @param op_name Name of the Op
  /// @return
  const std::vector<std::string> &
  get_op_outputs(const std::string &op_name) const {
    return MAP_AT(node_outputs_, op_name);
  }

  /// @brief Get producers of a tensor
  const std::vector<std::string> &
  get_producers(const std::string &tensor) const {
    return MAP_AT(producers_, tensor);
  }

  /// @brief Get consumers of a tensor
  const std::vector<std::string> &
  get_consumers(const std::string &tensor) const {
    return MAP_AT(consumers_, tensor);
  }

  /// @brief Get index of an op
  int get_op_index(const std::string &op) const {
    return static_cast<int>(MAP_AT(op_index_, op));
  }

  /// @brief Get number of ops in graph
  size_t get_num_ops() const { return meta_.op_list.size(); }

private:
  /// @brief Finds the input/output tensor names of each Op in the graph and
  /// cache it for later access.
  void fill_node_inputs_outputs() {
    for (const auto &op_info : meta_.op_list) {
      node_inputs_[op_info.name] = op_info.in_args;
      node_outputs_[op_info.name] = op_info.out_args;
    }
  }

  void fill_producers_consumers() {
    for (const auto &[tensor_name, tinfo] : meta_.tensor_map) {
      if (tinfo.parent_name != "const") {
        consumers_[tensor_name] = {};
        producers_[tensor_name] = {};
      }
    }

    for (const auto &[op, tensors] : node_inputs_) {
      for (const auto &tensor : tensors) {
        consumers_[tensor].push_back(op);
      }
    }

    for (const auto &[op, tensors] : node_outputs_) {
      for (const auto &tensor : tensors) {
        producers_[tensor].push_back(op);
      }
    }

    // Currently, one tensor has only one producer
    for (const auto &[tensor, producers] : producers_) {
      DD_ASSERT(producers.size() <= 1,
                OpsFusion::dd_format("{} has multiple producers", tensor));
    }
  }

  void fill_op_index() {
    const size_t n_ops = meta_.op_list.size();
    for (size_t i = 0; i < n_ops; ++i) {
      const auto &op = meta_.op_list.at(i);
      op_index_[op.name] = i;
    }
  }

private:
  OpsFusion::Metadata meta_;

  /// @brief Map of OpName --> Input tensor names
  std::map<std::string, std::vector<std::string>> node_inputs_;

  /// @brief Map of OpName --> Output tensor names
  std::map<std::string, std::vector<std::string>> node_outputs_;

  /// @brief Map of Tensor Name --> Producer Node
  std::map<std::string, std::vector<std::string>> producers_;

  /// @brief Map of Tensor Name --> Consumer Nodes
  std::map<std::string, std::vector<std::string>> consumers_;

  /// @brief Map of op --> OpIndex
  std::map<std::string, size_t> op_index_;
};

} // namespace detail
} // namespace Pass
} // namespace OpsFusion
