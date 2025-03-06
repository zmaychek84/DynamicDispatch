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

#include <any>
#include <iostream>
#include <vector>

#include <ops/op_interface.hpp>
#include <ops/record_timer/record_timer.hpp>
#include <utils/ipu_hw_config.hpp>
#include <utils/tfuncs.hpp>

#include <xaiengine.h>

namespace ryzenai {

record_timer::record_timer() {}

const std::vector<uint8_t> record_timer::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  // Get timer id
  uint32_t timer_id;
  if (attr.find("timer_id") != attr.end()) {
    timer_id = std::any_cast<uint32_t>(attr.find("timer_id")->second);
  } else {
    throw std::runtime_error("Can't find timer_id in attrs");
  }

  // Initialize AIE Driver. Hardcode for STRIX for now
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  record_timer_op_t timer_op;
  timer_op.id = timer_id;

  XAie_AddCustomTxnOp(&DevInst, XAIE_IO_CUSTOM_OP_RECORD_TIMER, &timer_op,
                      sizeof(timer_op));

  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&DevInst, 0, 0);
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_ptr;
  auto size = Hdr->TxnSize;

  std::vector<uint8_t> txn(size, 0);
  memcpy((void *)txn.data(), (void *)txn_ptr, size);

  // check if there is an API to free txn pointer
  free(txn_ptr);
  XAie_Finish(&DevInst);

  return txn;
}

} // namespace ryzenai
