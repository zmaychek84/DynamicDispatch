
/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <txn_container.hpp>
#include <utils/dpu_mdata.hpp>
#include <utils/instruction_registry.hpp>
#include <vector>

#include <ops/gelu_e/gelue.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

namespace ryzenai {

template class gelue<uint16_t, uint16_t>;

} // namespace ryzenai
