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

#include <unordered_set>

#include <op_fuser/fuse_types.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "cont_reuse.hpp"
#include "detail/graph_color.hpp"
#include "detail/meta_graph.hpp"
#include "passes.hpp"

/*
1. When this pass is called, it assumes that an initial buffer analysis is
already done and the buffer shapes for all scratch buffer is already computed in
meta.

2. As the name specifies, this optim is only for buffers in scratch space. Not
in buffers in input/output/const/super-param
*/

using namespace OpsFusion::Pass;
using namespace OpsFusion::Pass::detail;

static constexpr size_t TENSOR_PACK_ALIGNMENT = 4; // Bytes

using lifetable_t = std::vector<std::vector<node_t>>;

namespace OpsFusion {

/// @brief Assign an unique_id to each activation tensor in meta
static std::map<std::string, int> create_tensor_id_map(const Metadata &meta) {
  std::map<std::string, int> tensor_id_map;
  int id = 0;
  for (const auto &[t_name, t_info] : meta.tensor_map) {
    if (t_info.parent_name != "const") {
      tensor_id_map[t_name] = id++;
    }
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Tensor_id_map:\n{}", tensor_id_map));
  return tensor_id_map;
}

/// @brief Create an adjacency list notation of activation tensors
static AdjList
create_tensor_connectivity(const Metadata &meta,
                           const std::map<std::string, int> &tensor_id_map) {
  MetaGraph meta_graph(meta);

  AdjList adj_list;
  std::unordered_set<std::string> visited_tensors;
  for (const auto &op_info : meta.op_list) {
    const auto &op_inputs = meta_graph.get_op_inputs(op_info.name);
    const auto &op_outputs = meta_graph.get_op_outputs(op_info.name);
    for (const auto &op_output : op_outputs) {
      auto op_output_id = MAP_AT(tensor_id_map, op_output);
      if (visited_tensors.find(op_output) == visited_tensors.end()) {
        visited_tensors.insert(op_output);
        adj_list[op_output_id] = {};
      }
      for (const auto &op_input : op_inputs) {
        auto op_input_id = MAP_AT(tensor_id_map, op_input);
        adj_list[op_input_id].push_back(op_output_id);
      }
    }
  }

  RYZENAI_LOG_TRACE(dd_format("AdjList :\n{}", adj_list));

  return adj_list;
}

/// @brief Given an activation tensor and op execution schedule, this function
/// returns the lifetime of the tensor in terms of op indices.
/// A tensor's life starts at the op it produced and dies after execution of its
/// last consumer.
///
/// Example 1: if a tensor is produced by node_id:3 and it is consumed by 3
/// nodes (id:4, id:7, id:9), this returns {3, 9} because it is created by
/// node:3 while it has to be alive until node:9 is finished execution.
///
/// Example2 : If tensor is input tensor and consumed by node:5, its lifetime is
/// {0:5}
///
/// Example 3 : If tensor is output tensor and produced by node:5, its
/// lifetime is {5:N-1}, where N is the total number of ops
///
static std::pair<int, int> get_tensor_lifetime(const std::string &tensor_name,
                                               const MetaGraph &meta_graph) {
  std::pair<int, int> lifetime;

  const auto &producers = meta_graph.get_producers(tensor_name);
  std::vector<int> producer_ids;
  producer_ids.reserve(producers.size());
  for (const auto &producer : producers) {
    producer_ids.push_back(meta_graph.get_op_index(producer));
  }
  lifetime.first = producers.empty() ? 0
                                     : *std::min_element(producer_ids.begin(),
                                                         producer_ids.end());

  const size_t num_ops = meta_graph.get_num_ops();
  const auto &consumers = meta_graph.get_consumers(tensor_name);
  std::vector<int> consumer_ids;
  consumer_ids.reserve(consumers.size());
  for (const auto &consumer : consumers) {
    consumer_ids.push_back(meta_graph.get_op_index(consumer));
  }
  lifetime.second = consumers.empty() ? static_cast<int>(num_ops - 1)
                                      : *std::max_element(consumer_ids.begin(),
                                                          consumer_ids.end());

  DD_ASSERT(lifetime.first <= lifetime.second,
            OpsFusion::dd_format("Invalid Lifetime of tensor : {}, {}-{}",
                                 tensor_name, lifetime.first, lifetime.second));

  return lifetime;
}

/// @brief Given an op schedule and all its activation tensors, this function
/// returns a table in which each row corresponds to tensor ids live during each
/// op execution.
///
/// output is a 2D vector of size : [#ops x #tensor alive at that instance]
static lifetable_t
create_tensor_liveness_table(const Metadata &meta,
                             const std::map<std::string, int> &tensor_id_map) {
  MetaGraph meta_graph(meta);
  std::vector<std::vector<int>> live_table(meta.op_list.size(),
                                           std::vector<int>{});

  for (const auto &[tensor, id] : tensor_id_map) {
    auto lifetime = get_tensor_lifetime(tensor, meta_graph);
    for (size_t i = lifetime.first; i <= lifetime.second; ++i) {
      live_table[i].push_back(id);
    }
  }
  return live_table;
}

/// Given a life_table, this function cluster the tensors such that no two
/// tensors in a cluster are alive at same time. That means, tensors in a
/// cluster can share a common tensor.
// TODO : Can we remove I/O tensors directly from here instead of a separate
// step? Will that provide better results?
static std::map<node_t, ContReuse::Span>
group_on_liveness(const lifetable_t &life_table,
                  const std::map<int, std::string> &id_tensor_map,
                  const Metadata &meta,
                  ContReuse::BufferReuseAllocator &allocator) {
  std::map<node_t, ContReuse::Span> node_labels;

  // Assign mem for inputs
  const auto &new_ids = life_table.front();
  for (auto new_id : new_ids) {
    const auto &tname = MAP_AT(id_tensor_map, new_id);
    const auto &tinfo = MAP_AT(meta.tensor_map, tname);
    const auto &tsize = tinfo.size_in_bytes;
    auto size = Utils::align_to_next(tsize, TENSOR_PACK_ALIGNMENT);
    auto span = allocator.allocate(size);
    node_labels[new_id] = span;
  }

  // TODO : This would be slow impl. Revisit if required.
  for (size_t i = 1; i < life_table.size(); ++i) {
    const std::vector<node_t> &prev = life_table.at(i - 1);
    const std::vector<node_t> &curr = life_table.at(i);

    // Reclaim dead tensors' labels
    for (auto id : prev) {
      if (std::find(curr.begin(), curr.end(), id) == curr.end()) {
        allocator.deallocate(node_labels[id]);
      }
    }

    // Assign labels to new tensors
    for (auto id : curr) {
      if (std::find(prev.begin(), prev.end(), id) == prev.end()) {
        const auto &tname = MAP_AT(id_tensor_map, id);
        const auto &tinfo = MAP_AT(meta.tensor_map, tname);
        const auto &tsize = tinfo.size_in_bytes;
        auto size = Utils::align_to_next(tsize, TENSOR_PACK_ALIGNMENT);
        auto span = allocator.allocate(size);
        node_labels[id] = span;
      }
    }
  }

  return node_labels;
}

/// @brief Remove the graph I/O tensors from labels since they should not share
/// their buffers with any one.
static void
remove_io_from_labels(std::map<node_t, ContReuse::Span> &node_labels,
                      const std::map<std::string, node_t> &tensor_id_map,
                      const Metadata &meta) {
  RYZENAI_LOG_TRACE(dd_format("Labels :\n{}", node_labels));

  const auto &in_tensors = MAP_AT(meta.fused_tensors, "in").packed_tensors;
  for (const auto &tname : in_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    node_labels.erase(tid);
  }

  const auto &out_tensors = MAP_AT(meta.fused_tensors, "out").packed_tensors;
  for (const auto &tname : out_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    node_labels.erase(tid);
  }
}

/// @brief Just reverse node->label to label->[nodes]
static std::map<label_t, std::vector<node_t>>
node_labels_to_label_nodes(const std::map<node_t, label_t> &src) {
  std::map<label_t, std::vector<node_t>> dst;
  for (const auto &[key, value] : src) {
    dst[value] = {};
  }

  for (const auto &[key, value] : src) {
    dst[value].push_back(key);
  }

  RYZENAI_LOG_TRACE(dd_format("Label and Nodes :\n{}", dst));

  return dst;
}

/// @brief Reverse a 1:1 map
// TODO : This should go to common utils section
template <typename Key, typename Value>
static std::map<Value, Key> reverse_1to1_map(const std::map<Key, Value> &dict) {
  std::map<Value, Key> reverse_dict;
  for (const auto &[key, val] : dict) {
    reverse_dict[val] = key;
  }
  return reverse_dict;
}

/// @brief Compute the size of each bucket based on sizes of invidual tensor in
/// the bucket.
static std::map<label_t, size_t> compute_size_for_label(
    const std::map<label_t, std::vector<node_t>> &label_nodes,
    const std::map<int, std::string> &id_tensor_map, const Metadata &meta) {
  RYZENAI_LOG_TRACE("Computing Size for each label ... START");
  std::map<label_t, size_t> label_size;
  for (const auto &[label, tids] : label_nodes) {
    RYZENAI_LOG_TRACE(dd_format("  label : {}", label));
    size_t bucket_size = 0;
    for (auto tid : tids) {
      const auto &tname = MAP_AT(id_tensor_map, tid);
      const auto &tinfo = MAP_AT(meta.tensor_map, tname);
      bucket_size = std::max(bucket_size, tinfo.size_in_bytes);
      RYZENAI_LOG_TRACE(dd_format("    tid:{}, tensor:{}, size:{}", tid, tname,
                                  tinfo.size_in_bytes));
    }
    label_size[label] =
        Utils::align_to_next(bucket_size, TENSOR_PACK_ALIGNMENT);
  }

  RYZENAI_LOG_TRACE(dd_format("Label Size :\n{}", label_size));
  RYZENAI_LOG_TRACE("Computing Size for each label ... END");
  return label_size;
}

/// @brief Compute the cumulative size of all the buckets
static size_t compute_total_size(const std::map<label_t, size_t> &label_size,
                                 size_t alignment = TENSOR_PACK_ALIGNMENT) {
  size_t total_size = 0;
  for (const auto &[label, size] : label_size) {
    total_size += size;
  }
  return total_size;
}

/// @brief Compute the new offset for each bucket
static std::map<label_t, size_t>
compute_label_offsets(const std::map<label_t, size_t> &label_size) {
  std::map<label_t, size_t> label_offsets;
  size_t total_size = 0;
  for (const auto &[label, size] : label_size) {
    label_offsets[label] = total_size;
    total_size += size;
  }

  RYZENAI_LOG_TRACE(dd_format("label offsets: \n{}", label_offsets));
  return label_offsets;
}

/// @brief Update the tensor offsets in meta based on the new bucket offsets
static void update_meta_scratch_space(
    Metadata &meta, const std::map<std::string, node_t> &tensor_id_map,
    const std::map<node_t, ContReuse::Span> &node_labels, size_t total_size) {

  RYZENAI_LOG_TRACE("Patching meta scratch space ... START");

  const size_t max_tensor_padding_sz = meta.max_tensor_padding_sz;
  auto &scratch_buffer = MAP_AT(meta.fused_tensors, "scratch");

  scratch_buffer.size = Utils::align_to_next(total_size + max_tensor_padding_sz,
                                             TENSOR_PACK_ALIGNMENT);

  RYZENAI_LOG_TRACE(
      dd_format("Total Optimized Scratch Space : {}", scratch_buffer.size));

  for (const auto &tname : scratch_buffer.packed_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    auto span = MAP_AT(node_labels, tid);
    auto new_offset = span.start;
    auto &tinfo = MAP_AT(meta.tensor_map, tname);
    auto old_offset = tinfo.offset;
    // have a fixed offset to support padding of input tensors in scratch
    // assumes these are just used for alignment/data read patterns and
    // not used for computation
    tinfo.offset = new_offset + max_tensor_padding_sz;

    RYZENAI_LOG_TRACE(dd_format(
        "tid:{}, Span:[{}, {}], orig_offset:{} --> new_offset:{}", tid,
        span.start, span.end, old_offset, new_offset + max_tensor_padding_sz));
  }
  RYZENAI_LOG_TRACE("Patching meta scratch space ... END");
}

/// @brief validate the scratch space optimization
/// Check if tensors in the same label are alive at the same time
/// use life table & node_labels to validate this
static void validate_scratch_space_optimization(
    const lifetable_t &life_table,
    const std::map<node_t, ContReuse::Span> &node_labels) {
  RYZENAI_LOG_TRACE("Validating Scratch Space Optimization ... START");
  for (size_t i = 0; i < life_table.size(); ++i) {
    const std::vector<node_t> &tids = life_table.at(i);

    for (size_t j = 0; j < tids.size(); ++j) {
      auto tid = ARRAY_AT(tids, j);
      auto span1 = MAP_AT(node_labels, tid);
      for (size_t k = j + 1; k < tids.size(); ++k) {
        auto span2 = MAP_AT(node_labels, ARRAY_AT(tids, k));
        DD_ASSERT(
            !ContReuse::Span::intersection(span1, span2),
            OpsFusion::dd_format(
                "Scratch Space Optimization Failed. Multiple labels at op : {}",
                i));
      }
    }
  }
  RYZENAI_LOG_TRACE("Validating Scratch Space Optimization ... END");
}

void optimize_scratch_buffer_contiguous(Metadata &meta) {
  Metadata new_meta = remove_identity_ops(meta);
  RYZENAI_LOG_TRACE("Buffer Reuse Contiguous ... START");
  auto tensor_id_map = create_tensor_id_map(new_meta);
  auto id_tensor_map = reverse_1to1_map(tensor_id_map);
  auto life_table = create_tensor_liveness_table(new_meta, tensor_id_map);

  auto mem_view = std::make_unique<ContReuse::MemoryView>();
  ContReuse::BufferReuseAllocator allocator(std::move(mem_view));

  auto node_labels =
      group_on_liveness(life_table, id_tensor_map, new_meta, allocator);

  // Run validation before removing the io tensors
  validate_scratch_space_optimization(life_table, node_labels);

  remove_io_from_labels(node_labels, tensor_id_map, new_meta);

  auto total_scratch_size = allocator.size();
  update_meta_scratch_space(new_meta, tensor_id_map, node_labels,
                            total_scratch_size);
  set_new_offsets(meta, new_meta);
  RYZENAI_LOG_TRACE(MetaUtils::get_summary(meta));
  RYZENAI_LOG_TRACE("Buffer Reuse Contiguous ... END");
}

} // namespace OpsFusion
