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

#include <ops/op_interface.hpp>
#include <vector>

namespace ryzenai {
/**
 * @brief op_tiling_spec is a struct to store the tiling information,
 * it can be used for any op where tiling along any dimension require
 * the same operations, meaning that tiling is done wrt by the total
 * shape rather than the shape along any dimension
 */
struct op_tiling_spec {
  /**
   * @brief size_ is the total size of the op after tiling (sum of the sizes of
   * the tiles)
   */
  int64_t size_;
  /**
   * @brief cost_ is the total cost of the tiling
   */
  double cost_;
  /**
   * @brief info_ is a vector of pairs, where the first element is the size of
   * the tile and the second element is the shape of the tile
   */
  std::vector<std::pair<int64_t, std::vector<int64_t>>> info_;
  op_tiling_spec()
      : size_(0), cost_(0),
        info_(std::vector<std::pair<int64_t, std::vector<int64_t>>>()) {}
};

op_tiling_spec
map_padded_shape(int64_t M, int64_t K,
                 const std::vector<std::tuple<int, int>> &supported_shapes,
                 const std::map<int64_t, double> &tiling_cost);

std::vector<uint8_t> get_tiled_fused_txnbin(
    op_tiling_spec &tiling_spec,
    std::string (*get_instr_key)(std::string, size_t, size_t, int64_t),
    std::string txn_fname_prefix, std::vector<OpArgMap> &arg_map);

std::vector<uint8_t>
matmul_tile_transaction_bin(const std::vector<uint8_t> &base_txn_bin,
                            std::vector<OpArgMap> &args_map,
                            const std::vector<int64_t> &tile_info);

std::vector<uint8_t>
binary_op_tile_transaction_bin(const std::vector<uint8_t> &base_txn_bin,
                               std::vector<OpArgMap> &args_map,
                               const std::vector<size_t> &tile_info);

std::vector<uint8_t> binary_op_nonuniform_tile_transaction_bin(
    std::vector<std::vector<uint8_t>> &tiled_base_txn_bin,
    std::vector<OpArgMap> &args_map, const std::vector<int64_t> &tile_info);

std::vector<uint8_t> matmul_nonuniform_tile_transaction_bin(
    std::vector<std::vector<uint8_t>> &tiled_base_txn_bin,
    std::vector<OpArgMap> &args_map,
    const std::vector<std::vector<int64_t>> &tile_info);

std::pair<double, std::vector<int64_t>>
minimum_tiles(const std::set<int64_t> &tile,
              const std::map<int64_t, double> &cost, int64_t V);
} // namespace ryzenai
