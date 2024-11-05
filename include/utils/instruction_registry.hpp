/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#pragma once
#ifndef DYNAMIC_DISPATCH_UTILS_INSTRUCTION_REGISTRY_H
#define DYNAMIC_DISPATCH_UTILS_INSTRUCTION_REGISTRY_H

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <shared_mutex>

#include "logging.hpp"
#include <utils/instruction_cache.hpp>

namespace ryzenai {
namespace dynamic_dispatch {

class instruction_registry {
private:
  xrt::hw_context hw_ctx_;
  xrt::kernel kernel_;
  std::map<std::string, std::pair<bool, xrt::bo>> params_map_;
  std::shared_mutex instr_map_mutex_;
  std::shared_mutex params_map_mutex_;
  instruction_cache instr_cache_;

  void insert_to_instruction_map(std::pair<std::string, bool> &instr);
  void insert_to_layer_params_map(std::pair<std::string, bool> params);
  bool check_instr_in_registry(std::string key);

public:
  instruction_registry() = default;

  instruction_registry(const xrt::hw_context &hw_ctx,
                       const xrt::kernel &kernel);

  bool instr_in_registry(std::string key);

  void insert_fused_instr_to_instruction_map(
      std::pair<std::string, bool> &instr,
      const std::vector<uint8_t> &txn_bin_vec);

  void add_instructions(std::vector<std::pair<std::string, bool>> instr);

  void add_layer_params(std::vector<std::pair<std::string, bool>> params);

  xrt::bo get_instr_bo(std::string key);

  std::pair<bool, xrt::bo> get_param_bo(std::string key);
};

} // namespace dynamic_dispatch
} // namespace ryzenai

#endif /* DYNAMIC_DISPATCH_UTILS_INSTRUCTION_REGISTRY_H */
