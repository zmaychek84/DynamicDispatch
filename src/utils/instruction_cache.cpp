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

#include "utils/instruction_cache.hpp"
#include "txn/txn_utils.hpp"
#include <cstring>           // for std::memcpy
#include <txn_container.hpp> // for Transaction::getInstance()

namespace {

constexpr std::size_t INSTR_BO_LIMIT =
    60 * 1024 * 1024; // 64MB limit from XRT - 4MB buffer for PDI
constexpr std::size_t INSTR_BO_ALIGNMENT = 32 * 1024;

using ICACHE_NODE = std::tuple<std::string, std::size_t, xrt::bo>;
using ICACHE_NODE_LIST_ITERATOR = std::list<ICACHE_NODE>::iterator;
using ICACHE_NODE_LIST_REVERSE_ITERATOR =
    std::list<ICACHE_NODE>::reverse_iterator;

inline const std::string &get_key(const ICACHE_NODE_LIST_ITERATOR &it) {
  return std::get<0>(*it);
}

inline std::size_t get_size(const ICACHE_NODE_LIST_ITERATOR &it) {
  return std::get<1>(*it);
}

inline xrt::bo get_bo(const ICACHE_NODE_LIST_ITERATOR &it) {
  return std::get<2>(*it);
}

inline const std::string &
get_key(const ICACHE_NODE_LIST_REVERSE_ITERATOR &rit) {
  return std::get<0>(*rit);
}

inline std::size_t get_size(const ICACHE_NODE_LIST_REVERSE_ITERATOR &rit) {
  return std::get<1>(*rit);
}

inline xrt::bo get_bo(const ICACHE_NODE_LIST_REVERSE_ITERATOR &rit) {
  return std::get<2>(*rit);
}

} // namespace

namespace ryzenai {
namespace dynamic_dispatch {

instruction_cache::instruction_cache(const xrt::hw_context &hw_ctx,
                                     const xrt::kernel &kernel)
    : hw_ctx_(hw_ctx), kernel_(kernel), capacity_(INSTR_BO_LIMIT) {
  RYZENAI_LOG_TRACE("Constructing instruction cache");
}

bool instruction_cache::present(const std::string &key) {
  return map_.count(key) == std::size_t(1);
}

xrt::bo instruction_cache::get(const std::string &key) {
  if (present(key)) {
    touch(key);
    return get_bo(map_[key]);
  }

  put(key);

  return get_bo(map_[key]);
}

void instruction_cache::put(const std::string &key) {
  if (present(key)) {
    return;
  }

  const std::string &txn_string = Transaction::getInstance().get_txn_str(key);

  put(key, txn_string);
}

void instruction_cache::put(const std::string &key,
                            const std::string &txn_string) {
  if (present(key)) {
    return;
  }

  if (txn_string.size() < sizeof(XAie_TxnHeader)) {
    throw std::runtime_error(
        "Transaction string is too small to contain a valid header");
  }
  XAie_TxnHeader hdr;
  std::memcpy(&hdr, txn_string.data(), sizeof(XAie_TxnHeader));

  if (hdr.TxnSize != txn_string.size()) {
    throw std::runtime_error(
        "Transaction string is smaller than the size reported in the header.");
  }
  std::vector<uint8_t> txn_bin(txn_string.begin(), txn_string.end());
  auto i_buf = transaction_op(txn_bin);
  size_t instr_bo_size = i_buf.get_txn_instr_size();
  std::size_t instr_bo_aligned_size =
      Utils::align_to_next(instr_bo_size, INSTR_BO_ALIGNMENT);

  occupancy_ += instr_bo_aligned_size;

  while (occupancy_ > capacity_) {
    evict();
  }

  xrt::bo instr_bo;
  bool success = false;

  while (!success) {
    try {
      instr_bo = xrt::bo(hw_ctx_, instr_bo_size, xrt::bo::flags::cacheable,
                         kernel_.group_id(1));

      // If no exception is thrown, the operation is successful
      success = true;
    } catch (const std::exception &) {
      evict();
    }
  }

  instr_bo.write(i_buf.get_txn_op().data());
  instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  list_.push_front(std::make_tuple(key, instr_bo_aligned_size, instr_bo));
  map_[key] = list_.begin();

  RYZENAI_LOG_TRACE("[INSTR_CACHE] instr_bo created and saved key: " + key);
}

void instruction_cache::touch(const std::string &key) {
  if (present(key)) {
    list_.splice(list_.begin(), list_, map_[key]);
  }
}

void instruction_cache::evict() {
  if (!list_.empty()) {
    auto &key = get_key(list_.rbegin());
    RYZENAI_LOG_TRACE("[INSTR_CACHE] Evicting instr_bo: " + key);
    occupancy_ -= get_size(list_.rbegin());
    map_.erase(key);
    list_.pop_back();
  }
}

} // namespace dynamic_dispatch
} // namespace ryzenai
