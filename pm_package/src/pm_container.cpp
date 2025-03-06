// Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#include "pm_container.hpp"

std::vector<std::uint8_t> Preemption::get_pm_bvec(const std::string &name) {
  const std::string &txn_string = Preemption::get_pm_str(name);
  std::vector<std::uint8_t> txnData(txn_string.begin(), txn_string.end());
  return txnData;
}
