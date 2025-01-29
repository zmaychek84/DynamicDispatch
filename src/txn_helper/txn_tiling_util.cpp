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

#include "txn_tiling_util.hpp"
#include "txn/txn_utils.hpp"
#include <cfloat>
#include <txn_container.hpp>

using utils::txn_util;
namespace ryzenai {

/**
 * @brief finds nonuniform tiling shapes based on input shape
 * @param M: input shape M
 * @param K: input shape K
 * @param supported_shapes: list of supported shapes
 * @param tiling_cost: cost of each shape
 * @returns binary tiling specification
 */
op_tiling_spec
map_padded_shape(int64_t M, int64_t K,
                 const std::vector<std::tuple<int, int>> &supported_shapes,
                 const std::map<int64_t, double> &tiling_cost) {
  RYZENAI_LOG_TRACE("MAP PADDED SHAPE");
  op_tiling_spec tiling_spec;
  std::unordered_map<int64_t, std::vector<int64_t>> possible_tiles;
  std::set<int64_t> tiling;
  std::set<int64_t> K_shapes;
  for (const auto &supported : supported_shapes) {
    int64_t mat_M = std::get<0>(supported);
    int64_t mat_K = std::get<1>(supported);
    K_shapes.insert(mat_K);
    std::vector<int64_t> s = {mat_M, mat_K};
    int64_t shape_size = mat_M * mat_K;
    if ((mat_M == M && mat_K == K) || shape_size == M * K) {
      tiling_spec.size_ = shape_size;
      if (tiling_cost.count(shape_size)) {
        tiling_spec.cost_ = tiling_cost.at(shape_size);
      } else {
        tiling_spec.cost_ = 100;
      }
      tiling_spec.info_.push_back(std::make_pair(shape_size, s));
      return tiling_spec;
    }
    if (shape_size > M * K) {
      tiling_spec.size_ = shape_size;
      if (tiling_cost.count(shape_size)) {
        tiling_spec.cost_ = tiling_cost.at(shape_size);
      } else {
        tiling_spec.cost_ = 100;
      }
      tiling_spec.info_.push_back(std::make_pair(shape_size, s));
      return tiling_spec;
    }
    possible_tiles.insert({shape_size, s});
    tiling.insert(shape_size);
  }

  // limit the search space by removing M = 1
  if (tiling.size() > K_shapes.size()) {
    for (const int64_t &k : K_shapes) {
      if (tiling.find(k) != tiling.end()) {
        tiling.erase(k);
      }
    }
  }
  std::pair<double, std::vector<int64_t>> tiling_shape =
      minimum_tiles(tiling, tiling_cost, M * K);

  int64_t shape =
      std::reduce(tiling_shape.second.begin(), tiling_shape.second.end());
  tiling_spec.size_ = shape;
  tiling_spec.cost_ = tiling_shape.first;
  std::vector<std::pair<int64_t, std::vector<int64_t>>> tiling_info;
  for (auto s : tiling_shape.second) {
    tiling_info.push_back(std::make_pair(s, possible_tiles.at(s)));
    RYZENAI_LOG_TRACE("Tile: " + std::to_string(s) + " " +
                      std::to_string(possible_tiles.at(s).at(0)) + " " +
                      std::to_string(possible_tiles.at(s).at(1)));
  }
  tiling_spec.info_ = std::move(tiling_info);
  RYZENAI_LOG_TRACE("Tiling shape: " + std::to_string(shape));
  return tiling_spec;
}

/**
 * @brief Get the fused transaction binary based on the tiling specification
 * @param tiling_spec: tiling specification
 * @param get_instr_key: function to get the instruction key
 * @param arg_map: op arg map
 * @returns fused transaction binary
 */
std::vector<uint8_t> get_tiled_fused_txnbin(
    op_tiling_spec &tiling_spec,
    std::string (*get_instr_key)(std::string, size_t, size_t, int64_t),
    std::string txn_fname_prefix, std::vector<OpArgMap> &arg_map) {
  std::vector<uint8_t> data;
  Transaction &txn = Transaction::getInstance();
  std::vector<int64_t> tile_patch_info;
  tile_patch_info.reserve(tiling_spec.info_.size());
  std::vector<std::vector<uint8_t>> tiling_txn_bin;
  for (auto const &info : tiling_spec.info_) {
    // should match the xrt arg index of each op
    int64_t shape = info.first;
    std::vector<int64_t> shape_info = info.second;
    tile_patch_info.emplace_back(shape);
    std::string txn_key;
    if (shape_info.size() > 1) {
      txn_key = get_instr_key(txn_fname_prefix, shape_info.at(0),
                              shape_info.at(1), -1);
    } else {
      txn_key = get_instr_key(txn_fname_prefix, 0, 0, shape);
    }
    tiling_txn_bin.push_back(txn.get_txn_bvec(txn_key));
  }
  data = binary_op_nonuniform_tile_transaction_bin(tiling_txn_bin, arg_map,
                                                   tile_patch_info);
  return data;
}

double minimum_tiles_helper(
    const std::set<int64_t> &tile, const std::map<int64_t, double> &cost,
    int64_t V,
    std::unordered_map<int64_t, std::pair<double, std::vector<int64_t>>>
        &memo) {
  if (V <= 0) {
    return 0;
  }
  // Initialize result
  double res = DBL_MAX;
  if (memo.count(V)) {
    return memo[V].first;
  }
  std::vector<int64_t> minSolution;

  for (int64_t m : tile) {
    double sub_res = minimum_tiles_helper(tile, cost, V - m, memo);
    double m_cost;
    try {
      m_cost = cost.at(m);
    } catch (...) {
      m_cost = 100;
    }
    if (sub_res >= 0 && sub_res + (m_cost) + 0.5 < res) {
      res = sub_res + (m_cost) + 0.5;
      minSolution = memo[V - m].second;
      minSolution.push_back(m);
    }
  }
  memo[V] = {res == DBL_MAX ? -1 : res, minSolution};
  return memo[V].first;
}

/**
 * @brief finds nonuniform tiling parameters based on the vector of tiles and
 * cost function
 *
 * @param tile: list of supported tile shapes
 * @param cost: cost function for each tile shape
 * @param V: target shape
 *
 * @returns: list of tiles for the target shape
 */
std::pair<double, std::vector<int64_t>>
minimum_tiles(const std::set<int64_t> &tile,
              const std::map<int64_t, double> &cost, int64_t V) {
  std::unordered_map<int64_t, std::pair<double, std::vector<int64_t>>> memo;
  double res = minimum_tiles_helper(tile, cost, V, memo);
  if (res < 0) {
    DD_THROW("No valid tiling found");
  }
  return memo[V];
}

/**
 * @brief Tile the transaction binary of matmul based on the tiling specified by
 * tile info
 *
 * @param base_txn_bin: the kernel transaction bin
 * @param args_map: contains the buffer sizes of the base kernel operation
 * @param tile_info: number of tiles along each dimension
 *
 * @returns fused transaction bin based on the tiling scheme
 */
std::vector<uint8_t>
matmul_tile_transaction_bin(const std::vector<uint8_t> &base_txn_bin,
                            std::vector<OpArgMap> &args_map,
                            const std::vector<int64_t> &tile_info) {
  int64_t tile_M = tile_info.at(0);
  int64_t tile_K = tile_info.at(1);
  int64_t tile_N = tile_info.at(2);
  RYZENAI_LOG_TRACE("Tiling matmul transaction bin with tile_info: M: " +
                    std::to_string(tile_M) + ", K: " + std::to_string(tile_K) +
                    ", N: " + std::to_string(tile_N));
  DD_THROW_IF((tile_K != 1),
              "Tiling matrix along k dimension not supported yet");

  // TODO: tiling along other dimensions?
  std::vector<std::vector<uint8_t>> txn_vecs;

  txn_vecs.reserve(tile_M * tile_K * tile_N);
  txn_vecs.push_back(base_txn_bin);

  for (int64_t m = 0; m < tile_M; m++) {
    for (int64_t n = 0; n < tile_N; n++) {
      if (m == 0 && n == 0) {
        continue;
      }
      for (auto it = args_map.begin(); it != args_map.end(); ++it) {
        switch (it->arg_type) {
        case OpArgMap::OpArgType::INPUT: {
          it->offset = m * it->size;
        } break;
        case OpArgMap::OpArgType::CONST_INPUT: {
          it->offset = n * it->size;
        } break;
        case OpArgMap::OpArgType::OUTPUT: {
          it->offset += it->size;
        } break;
        }
      }
      std::vector<uint8_t> patched_tile_txn =
          txn_util::patch(base_txn_bin, args_map);
      txn_vecs.push_back(std::move(patched_tile_txn));
    }
  }
  auto fused_txn = txn_util::fuse_txns(txn_vecs);
  return fused_txn;
}

/**
 * @brief Tile the transaction binary of elewadd based on the tiling specified
 * by tile info
 *
 * @param base_txn_bin: the kernel transaction bin
 * @param args_map: contains the buffer sizes of the base kernel operation
 * @param tile_info: number of tiles total
 *
 * @returns fused transaction bin based on the tiling scheme
 */
std::vector<uint8_t>
binary_op_tile_transaction_bin(const std::vector<uint8_t> &base_txn_bin,
                               std::vector<OpArgMap> &args_map,
                               const std::vector<size_t> &tile_info) {
  int64_t tile = tile_info.at(0);
  RYZENAI_LOG_TRACE("Tiling binary transaction bin with tile_info: " +
                    std::to_string(tile));
  std::vector<std::vector<uint8_t>> txn_vecs;
  txn_vecs.reserve(tile);
  txn_vecs.push_back(base_txn_bin);

  for (int64_t t = 1; t < tile; t++) {
    for (auto it = args_map.begin(); it != args_map.end(); ++it) {
      switch (it->arg_type) {
      case OpArgMap::OpArgType::INPUT: {
        it->offset += it->size;
      } break;
      case OpArgMap::OpArgType::OUTPUT: {
        it->offset += it->size;
      } break;
      }
    }
    std::vector<uint8_t> patched_tile_txn =
        txn_util::patch(base_txn_bin, args_map);
    txn_vecs.push_back(std::move(patched_tile_txn));
  }
  auto fused_txn = txn_util::fuse_txns(txn_vecs);
  return fused_txn;
}

/**
 * @brief Tile the transaction binary of matmul based on the tiling specified.
 * Each tile coul be not of the same size, the size and sequence information is
 * specified in tile info.
 *
 * @param base_txn_bin: the kernel transaction bin list for different shapes, in
 * the sequence of tile info.
 * @param args_map: contains the buffer sizes of the base (unit) kernel
 * operation
 * @param tile_info: a vector of size 3 specifying the tiling information, first
 * dimension is tiling info for M and so on
 *
 * @returns fused transaction bin based on the tiling scheme
 */
std::vector<uint8_t> matmul_nonuniform_tile_transaction_bin(
    std::vector<std::vector<uint8_t>> &tiled_base_txn_bin,
    std::vector<OpArgMap> &args_map,
    const std::vector<std::vector<int64_t>> &tile_info) {
  RYZENAI_LOG_TRACE("Tiling matmul nonuniform transaction bin");
  std::vector<int64_t> tile_M = tile_info.at(0);
  std::vector<int64_t> tile_K = tile_info.at(1);
  std::vector<int64_t> tile_N = tile_info.at(2);
  DD_THROW_IF((tile_K.size() != 0),
              "Tiling matrix along k dimension not supported yet");
  DD_THROW_IF((tile_N.size() != 0),
              "Tiling matrix along n dimension not supported yet");
  for (size_t m = 1; m < tile_M.size(); m++) {
    int64_t M = tile_M.at(m - 1);
    for (auto it = args_map.begin(); it != args_map.end(); ++it) {
      switch (it->arg_type) {
      case OpArgMap::OpArgType::INPUT: {
        it->offset += M * it->size;
      } break;
      case OpArgMap::OpArgType::OUTPUT: {
        it->offset += M * it->size;
      } break;
      case OpArgMap::OpArgType::SCRATCH_PAD: {
        it->offset += M * it->size;
      } break;
      }
    }
    tiled_base_txn_bin.at(m) =
        txn_util::patch(tiled_base_txn_bin.at(m), args_map);
  }

  auto fused_txn = txn_util::fuse_txns(tiled_base_txn_bin);
  return fused_txn;
}

/**
 * @brief Tile the transaction binary of rmsnorm based on the tiling specified.
 * Each tile coul be not of the same size, the size and sequence information is
 * specified in dest arg maps.
 *
 * @param base_txn_bin: the kernel transaction bin list for different shapes, in
 * the sequence of dest arg maps.
 * @param source_arg_map: contains the buffer sizes of the base (unit) kernel
 * operation
 * @param dest_arg_maps: a vector of vector of OpArgMap specifying the tiling
 * information for each tile
 *
 * @returns fused transaction bin based on the tiling scheme
 */
std::vector<uint8_t> rmsnorm_nonuniform_tile_transaction_bin(
    std::vector<std::vector<uint8_t>> &tiled_base_txn_bin,
    const std::vector<OpArgMap> &source_arg_map,
    const std::vector<std::vector<OpArgMap>> &dest_arg_maps) {
  RYZENAI_LOG_TRACE("Tiling matmul nonuniform transaction bin");
  DD_THROW_IF((tiled_base_txn_bin.size() != dest_arg_maps.size()),
              "base txn bin and dest arg maps size mismatch");
  DD_THROW_IF((tiled_base_txn_bin.size() < 1),
              "base txn bin size should be at least 1");
  for (int i = 1; i < tiled_base_txn_bin.size(); i++) {
    tiled_base_txn_bin.at(i) = txn_util::patch(
        tiled_base_txn_bin.at(i), source_arg_map, dest_arg_maps.at(i));
  }
  auto fused_txn = txn_util::fuse_txns(tiled_base_txn_bin);
  return fused_txn;
}

/**
 * @brief Patch the transaction binary of binary operators based on the
 * nonuniform tiling specified by the tile info
 *
 * @param base_txn_bin: the kernel transaction bin list for different shapes, in
 * the sequence of tile info.
 * @param args_map: contains the op arg map of the op, size is ignored
 * @param tile_info: the list of tile shapes to be tiled
 */
std::vector<uint8_t> binary_op_nonuniform_tile_transaction_bin(
    std::vector<std::vector<uint8_t>> &tiled_base_txn_bin,
    std::vector<OpArgMap> &args_map, const std::vector<int64_t> &tile_info) {
  RYZENAI_LOG_TRACE("Tiling binary op nonuniform transaction bin");

  for (size_t t = 1; t < tile_info.size(); ++t) {
    for (auto it = args_map.begin(); it != args_map.end(); ++it) {
      it->offset += tile_info.at(t - 1) * (it->size);
    }
    tiled_base_txn_bin.at(t) =
        txn_util::patch(tiled_base_txn_bin.at(t), args_map);
  }

  auto fused_txn = txn_util::fuse_txns(tiled_base_txn_bin);
  return fused_txn;
}
} // namespace ryzenai
