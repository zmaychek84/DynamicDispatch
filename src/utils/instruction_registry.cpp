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

#include "utils/instruction_registry.hpp"
#include <txn_container.hpp>

namespace ryzenai {
namespace dynamic_dispatch {

instruction_registry::instruction_registry(const xrt::hw_context &hw_ctx,
                                           const xrt::kernel &kernel)
    : hw_ctx_(hw_ctx), kernel_(kernel), instr_cache_(hw_ctx, kernel) {

  RYZENAI_LOG_TRACE("Constructing instruction registry");
}

void instruction_registry::insert_to_instruction_map(
    std::pair<std::string, bool> &instr) {
  instr_cache_.put(instr.first);
}

void instruction_registry::insert_to_layer_params_map(
    std::pair<std::string, bool> params) {

  std::string layer_params =
      Transaction::getInstance().get_txn_str(params.first);
  std::vector<char> prm_buffer(layer_params.begin(), layer_params.end());
  size_t prm_size = prm_buffer.size();
  xrt::bo param_bo = xrt::bo(hw_ctx_, prm_size, xrt::bo::flags::host_only,
                             kernel_.group_id(8));
  param_bo.write(prm_buffer.data());
  param_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  RYZENAI_LOG_TRACE("[INSTR_REG] instr_bo created and saved: " + params.first);

  params_map_.insert({params.first, std::make_pair(params.second, param_bo)});
}

bool instruction_registry::check_instr_in_registry(std::string key) {
  return instr_cache_.present(key);
}

bool instruction_registry::instr_in_registry(std::string key) {
  std::shared_lock<std::shared_mutex> r_guard(instr_map_mutex_);
  return check_instr_in_registry(key);
}

void instruction_registry::insert_fused_instr_to_instruction_map(
    std::pair<std::string, bool> &instr,
    const std::vector<uint8_t> &txn_bin_vec) {
  std::lock_guard<std::shared_mutex> w_guard(instr_map_mutex_);
  std::string txn_string(txn_bin_vec.begin(), txn_bin_vec.end());
  instr_cache_.put(instr.first, txn_string);
}

void instruction_registry::add_instructions(
    std::vector<std::pair<std::string, bool>> instr) {}

void instruction_registry::add_layer_params(
    std::vector<std::pair<std::string, bool>> params) {
  std::lock_guard<std::shared_mutex> w_guard(params_map_mutex_);
  for (auto &i : params) {
    insert_to_layer_params_map(i);
  }
}

xrt::bo instruction_registry::get_instr_bo(std::string key) {
  std::lock_guard<std::shared_mutex> w_guard(instr_map_mutex_);
  RYZENAI_LOG_TRACE("Getting instruction key: " + key);
  return instr_cache_.get(key);
}

std::pair<bool, xrt::bo> instruction_registry::get_param_bo(std::string key) {
  std::shared_lock<std::shared_mutex> r_guard(params_map_mutex_);
  auto val = params_map_.find(key);
  if (val == params_map_.end()) {
    throw std::runtime_error("Failed to get instruction buffer for key: " +
                             key);
  }
  return val->second;
}

} // namespace dynamic_dispatch
} // namespace ryzenai
