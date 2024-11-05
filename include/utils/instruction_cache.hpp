#pragma once
#ifndef DYNAMIC_DISPATCH_UTILS_INSTRUCTION_CACHE_HPP
#define DYNAMIC_DISPATCH_UTILS_INSTRUCTION_CACHE_HPP

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstddef>
#include <list>
#include <string>
#include <tuple>
#include <unordered_map>

#include <utils/logging.hpp>
#include <utils/utils.hpp>
#include <xaiengine.h>

namespace ryzenai {
namespace dynamic_dispatch {

// LRU Cache
class instruction_cache {
public:
  instruction_cache() = default;

  instruction_cache(const xrt::hw_context &hw_ctx, const xrt::kernel &kernel);

  bool present(const std::string &key);

  xrt::bo get(const std::string &key);

  void put(const std::string &key);

  void put(const std::string &key, const std::string &txn_string);

private:
  xrt::hw_context hw_ctx_;
  xrt::kernel kernel_;
  std::size_t capacity_;
  std::size_t occupancy_ = 0;
  std::list<std::tuple<std::string, std::size_t, xrt::bo>>
      list_; // list of ICACHE_NODE
  std::unordered_map<
      std::string,
      std::list<std::tuple<std::string, std::size_t, xrt::bo>>::iterator>
      map_;

  void touch(const std::string &key);

  void evict();
};

} // namespace dynamic_dispatch
} // namespace ryzenai

#endif
