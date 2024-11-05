// Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#include "txn_container.hpp"

std::vector<std::uint8_t> Transaction::get_txn_bvec(const std::string &name) {
  const std::string &txn_string = Transaction::get_txn_str(name);
  std::vector<std::uint8_t> txnData(txn_string.begin(), txn_string.end());
  return txnData;
}
