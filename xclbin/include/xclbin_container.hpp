// Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

#ifndef XCLBIN_CONTAINER_H
#define XCLBIN_CONTAINER_H

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

class XclbinContainer {
private:
  XclbinContainer();

public:
  static XclbinContainer &getInstance() {
    static XclbinContainer instance;
    return instance;
  }

  XclbinContainer(XclbinContainer const &) = delete;

  void operator=(XclbinContainer const &) = delete;

  const std::vector<char> &get_xclbin_content(const std::string &name);
};
#endif
