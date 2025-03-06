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

#include "detail/graph_color.hpp"
#include "detail/meta_graph.hpp"
#include "passes.hpp"

/*
1. When this pass is called, it assumes that an initial buffer analysis is
already done and the buffer shapes for all scratch buffer is already computed in
meta.

2. As the name specifies, this optim is only for buffers in scratch space. Not
in buffers in input/output/const/super-param

3. TODO : With current meta structure, it is hard to differentiate b/w input &
output tensors of an op from the meta itself. Changing meta structure might need
a larger change in both DD & vaip. So for now, this impl will rely on
op->get_buff_reqs() to identify inputs & outputs.

There are mutliple better solutions here.
    3.a. (P0) Differentiate ip & op at meta level itself.
    3.b. (P1) Create non-DAG graph of tensor connections.
*/

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

struct LifeInfo {
  size_t size = 0;
  size_t nbuckets = 0;
};

static std::vector<LifeInfo>
total_size_at_each_op(const lifetable_t &life_table,
                      const std::map<int, std::string> &id_tensor_map,
                      const Metadata &meta) {
  std::vector<LifeInfo> total_size(meta.op_list.size());
  for (size_t i = 0; i < life_table.size(); ++i) {
    const std::vector<int> &tids = life_table.at(i);
    total_size[i].nbuckets = tids.size();
    for (auto tid : tids) {
      const auto &tname = MAP_AT(id_tensor_map, tid);
      const auto &tinfo = MAP_AT(meta.tensor_map, tname);
      total_size[i].size += tinfo.size_in_bytes;
    }
  }
  return total_size;
}

static void
print_total_size_at_each_op(const std::vector<LifeInfo> &total_size) {
  std::cout << "Total Size at each op : " << std::endl;
  // for (size_t i = 0; i < total_size.size(); ++i) {
  //   std::cout << "  Op:" << i << " Size (B):" << total_size[i] << std::endl;
  // }
  auto size_iter = std::max_element(
      total_size.begin(), total_size.end(),
      [](const LifeInfo &a, const LifeInfo &b) { return a.size < b.size; });
  auto bucket_iter = std::max_element(total_size.begin(), total_size.end(),
                                      [](const LifeInfo &a, const LifeInfo &b) {
                                        return a.nbuckets < b.nbuckets;
                                      });
  std::cout << "Max size : " << size_iter->size
            << ", nbuckets : " << size_iter->nbuckets
            << ", at op : " << std::distance(total_size.begin(), size_iter)
            << std::endl;
  std::cout << "Max Buckets : " << bucket_iter->nbuckets
            << ", size : " << bucket_iter->size
            << ", at op : " << std::distance(total_size.begin(), bucket_iter)
            << std::endl;
}

/// @brief Compute the absolute difference of two uint64_t values safely
static uint64_t safe_diff(uint64_t a, uint64_t b) {
  return a > b ? a - b : b - a;
}

/// @brief convert a uint64_t value to int64_t with exception if overflow
static int64_t safe_uint64_to_int64(uint64_t val) {
  if (val > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("Value is too large to fit in int64_t");
  }
  return static_cast<int64_t>(val);
}

/// @brief Given a set of labels, this function returns an unused label from the
/// set. If the set is empty, it returns a fresh label.
/// Otherwise, it returns a label from the set such that the change in total
/// size of tensor in each label is minimized.
/// @param unused_labels : Set of labels which are free to use.
/// @param label_size : Memory size of each label/group
static label_t get_label_with_minimal_size_change(
    const label_t curr_label, std::set<label_t> &unused_labels,
    std::map<label_t, size_t> &label_size,
    const std::map<int, std::string> &id_tensor_map, const Metadata &meta) {
  const auto &tname = MAP_AT(id_tensor_map, curr_label);
  const auto &tinfo = MAP_AT(meta.tensor_map, tname);
  const auto &curr_label_size = tinfo.size_in_bytes;

  if (unused_labels.empty()) {
    auto new_label = static_cast<label_t>(label_size.size());
    label_size[new_label] = curr_label_size;
    return new_label;
  }

  int64_t min_size_delta = std::numeric_limits<int64_t>::max();
  label_t min_label = 0;
  for (auto label : unused_labels) {
    int64_t curr_size_delta = safe_uint64_to_int64(curr_label_size) -
                              safe_uint64_to_int64(label_size.at(label));
    curr_size_delta = std::abs(curr_size_delta);
    if (curr_size_delta < min_size_delta) {
      min_size_delta = curr_size_delta;
      min_label = label;
    }
  }

  label_size[min_label] = std::max(label_size[min_label], curr_label_size);
  unused_labels.erase(min_label);

  return min_label;
}

/// Given a life_table, this function cluster the tensors such that no two
/// tensors in a cluster are alive at same time. That means, tensors in a
/// cluster can share a common tensor.
// TODO : Can we remove I/O tensors directly from here instead of a separate
// step? Will that provide better results?
static std::map<node_t, label_t>
group_on_liveness(const lifetable_t &life_table,
                  const std::map<int, std::string> &id_tensor_map,
                  const Metadata &meta) {
  label_t group_id = 0;
  std::stack<label_t> free_group_ids;
  std::map<node_t, label_t> node_labels;
  std::map<label_t, size_t> label_size;

  // As a start, Assign fresh labels to input tensors
  const auto &new_ids = life_table.front();
  for (auto new_id : new_ids) {
    auto gid = Pass::detail::get_unused_label(free_group_ids, group_id);
    node_labels[new_id] = gid;
  }

  // TODO : This would be slow impl. Revisit if required.
  for (size_t i = 1; i < life_table.size(); ++i) {
    const std::vector<node_t> &prev = life_table.at(i - 1);
    const std::vector<node_t> &curr = life_table.at(i);

    // Reclaim dead tensors' labels
    for (auto id : prev) {
      if (std::find(curr.begin(), curr.end(), id) == curr.end()) {
        free_group_ids.push(MAP_AT(node_labels, id));
      }
    }

    // Assign labels to new tensors
    for (auto id : curr) {
      if (std::find(prev.begin(), prev.end(), id) == prev.end()) {
        auto gid = get_unused_label(free_group_ids, group_id);
        node_labels[id] = gid;
      }
    }
  }
  return node_labels;
}

static std::map<node_t, label_t>
group_on_liveness_v2(const lifetable_t &life_table,
                     const std::map<int, std::string> &id_tensor_map,
                     const Metadata &meta) {
  label_t group_id = 0;
  std::set<label_t> free_group_ids;
  std::map<node_t, label_t> node_labels;
  std::map<label_t, size_t> label_size;

  // As a start, Assign fresh labels to input tensors
  const auto &new_ids = life_table.front();
  for (auto new_id : new_ids) {
    auto gid = get_label_with_minimal_size_change(
        new_id, free_group_ids, label_size, id_tensor_map, meta);
    node_labels[new_id] = gid;
  }

  // TODO : This would be slow impl. Revisit if required.
  for (size_t i = 1; i < life_table.size(); ++i) {
    const std::vector<node_t> &prev = life_table.at(i - 1);
    const std::vector<node_t> &curr = life_table.at(i);

    // Reclaim dead tensors' labels
    for (auto id : prev) {
      if (std::find(curr.begin(), curr.end(), id) == curr.end()) {
        free_group_ids.insert(MAP_AT(node_labels, id));
      }
    }

    // Assign labels to new tensors
    for (auto id : curr) {
      if (std::find(prev.begin(), prev.end(), id) == prev.end()) {
        auto gid = get_label_with_minimal_size_change(
            id, free_group_ids, label_size, id_tensor_map, meta);
        node_labels[id] = gid;
      }
    }
  }
  return node_labels;
}

static void
remove_io_from_labels(std::map<node_t, label_t> &node_labels,
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

  RYZENAI_LOG_TRACE(dd_format("Labels after removing IO:\n{}", node_labels));
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
    const std::map<node_t, label_t> &node_labels,
    const std::map<label_t, size_t> &label_offsets, size_t total_size) {

  RYZENAI_LOG_TRACE("Patching meta scratch space ... START");

  const size_t max_tensor_padding_sz = meta.max_tensor_padding_sz;
  auto &scratch_buffer = MAP_AT(meta.fused_tensors, "scratch");

  scratch_buffer.size = Utils::align_to_next(total_size + max_tensor_padding_sz,
                                             TENSOR_PACK_ALIGNMENT);

  RYZENAI_LOG_TRACE(
      dd_format("Total Optimized Scratch Space : {}", scratch_buffer.size));

  for (const auto &tname : scratch_buffer.packed_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    auto label = MAP_AT(node_labels, tid);
    auto new_offset = MAP_AT(label_offsets, label);
    auto &tinfo = MAP_AT(meta.tensor_map, tname);
    auto old_offset = tinfo.offset;
    // have a fixed offset to support padding of input tensors in scratch
    // assumes these are just used for alignment/data read patterns and
    // not used for computation
    tinfo.offset = new_offset + max_tensor_padding_sz;

    RYZENAI_LOG_TRACE(
        dd_format("tid:{}, label:{}, orig_offset:{} --> new_offset:{}", tid,
                  label, old_offset, new_offset + max_tensor_padding_sz));
  }
  RYZENAI_LOG_TRACE("Patching meta scratch space ... END");
}

/// @brief validate the scratch space optimization
/// Check if tensors in the same label are alive at the same time
/// use life table & node_labels to validate this
static void validate_scratch_space_optimization(
    const lifetable_t &life_table,
    const std::map<node_t, label_t> &node_labels) {
  RYZENAI_LOG_TRACE("Validating Scratch Space Optimization ... START");
  for (size_t i = 0; i < life_table.size(); ++i) {
    const std::vector<node_t> &tids = life_table.at(i);
    std::set<label_t> labels;
    for (auto tid : tids) {
      labels.insert(MAP_AT(node_labels, tid));
    }
    DD_ASSERT(labels.size() == tids.size(),
              OpsFusion::dd_format("Validation of Scratch Space Optimization "
                                   "failed. Multiple labels at op : {}",
                                   i));
  }
  RYZENAI_LOG_TRACE("Validating Scratch Space Optimization ... END");
}

void optimize_scratch_buffer_bucket(Metadata &meta,
                                    const std::string &buffer_reuse_version) {
  Metadata new_meta = remove_identity_ops(meta);
  RYZENAI_LOG_TRACE("Buffer Reuse via Buckets ... START");
  auto tensor_id_map = create_tensor_id_map(new_meta);
  auto id_tensor_map = reverse_1to1_map(tensor_id_map);
  auto life_table = create_tensor_liveness_table(new_meta, tensor_id_map);

  std::map<node_t, label_t> node_labels;
  if (buffer_reuse_version == "v1") {
    node_labels = group_on_liveness(life_table, id_tensor_map, new_meta);
  } else {
    node_labels = group_on_liveness_v2(life_table, id_tensor_map, new_meta);
  }

  // Validate the optimization before removing I/O tensors
  validate_scratch_space_optimization(life_table, node_labels);

  remove_io_from_labels(node_labels, tensor_id_map, new_meta);
  auto label_nodes = node_labels_to_label_nodes(node_labels);
  auto total_size = total_size_at_each_op(life_table, id_tensor_map, new_meta);
  // print_total_size_at_each_op(total_size);
  auto label_size =
      compute_size_for_label(label_nodes, id_tensor_map, new_meta);
  auto total_scratch_size = compute_total_size(label_size);
  auto label_offsets = compute_label_offsets(label_size);
  update_meta_scratch_space(new_meta, tensor_id_map, node_labels, label_offsets,
                            total_scratch_size);
  set_new_offsets(meta, new_meta);
  RYZENAI_LOG_TRACE(MetaUtils::get_summary(meta));
  RYZENAI_LOG_TRACE("Buffer Reuse via Buckets ... END");
}

} // namespace OpsFusion
