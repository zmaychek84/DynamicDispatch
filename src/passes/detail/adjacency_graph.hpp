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

#include <map>

namespace OpsFusion {
namespace Pass {
namespace detail {

using node_t = int;
using AdjList = std::map<node_t, std::vector<node_t>>;

// Convert an AdjList from "node -> child_nodes" to "node -> parent_nodes"
static AdjList child_graph_to_parent_graph(const AdjList &child_graph) {
  AdjList parent_graph;
  for (const auto &[key, value] : child_graph) {
    parent_graph[key] = {};
  }

  for (const auto &[node, children] : child_graph) {
    for (auto child : children) {
      parent_graph[child].push_back(node);
    }
  }

  return parent_graph;
}

class AdjGraph {
public:
  AdjGraph() = default;
  AdjGraph(AdjList child_graph) : child_graph_(std::move(child_graph)) {
    parent_graph_ = child_graph_to_parent_graph(child_graph_);
  }

  // Get Graph Inputs : nodes with no parents
  std::vector<node_t> get_graph_inputs() const {
    std::vector<node_t> inputs;
    for (const auto &[node, parents] : parent_graph_) {
      if (parents.empty()) {
        inputs.push_back(node);
      }
    }
    return inputs;
  }

  const std::vector<node_t> &get_children(node_t node) const {
    return child_graph_.at(node);
  }

  const std::vector<node_t> &get_parents(node_t node) const {
    return parent_graph_.at(node);
  }

private:
  AdjList child_graph_;
  AdjList parent_graph_;
};

} // namespace detail
} // namespace Pass
} // namespace OpsFusion
