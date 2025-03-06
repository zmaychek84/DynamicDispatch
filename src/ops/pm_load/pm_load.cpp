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

#include "pm_container.hpp"
#include <any>
#include <iostream>
#include <ops/op_interface.hpp>
#include <ops/pm_load/pm_load.hpp>
#include <utils/tfuncs.hpp>
#include <vector>

#include "txn/txn_utils.hpp"
#include "utils/dpu_mdata.hpp"
#include "utils/ipu_hw_config.hpp"

#include <xaiengine.h>

namespace ryzenai {

pm_load::pm_load(bool load_xrt) {
  std::string XCLBIN_FNAME = OpInterface::get_dd_base_dir() + "\\xclbin\\" +
                             "stx" + "\\square_cube.xclbin";
  PM_PREFIX = "aie_elf_ctrl_pkt_";
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  }
}

void pm_load::update_meta(const OpsFusion::OpPMMap &op_pm_map,
                          const OpsFusion::OverlayPMMeta &overlay_meta) {
  if (op_pm_map.op_to_pm_bin_map.empty() ||
      op_pm_map.pm_bin_to_meta_map.empty() ||
      overlay_meta.pkt_sw_meta.empty()) {
    return;
  }
  op_pm_map_ = op_pm_map;
  overlay_meta_ = overlay_meta;
}

OpsFusion::OverlayPMMeta pm_load::overlay_meta_ = {
    // TODO: Add nother field to capture PDI info if needed.
    4,
    {
        {
            0, 0, 0, 1, // pkt_id, col, dma_ch_num, num_cores
        },
    },
};

OpsFusion::OpPMMap pm_load::op_pm_map_ = {
    {
        {"cube", "PM_0_cube.bin"},
        {"square", "PM_1_sq.bin"},
    },
    {
        {"PM_0_cube.bin", {0, {3056}, {0}}},
        {"PM_1_sq.bin", {0, {3008}, {0}}},
    }};

const std::string pm_load::get_op_pmbin_name(const std::string &op_name,
                                             const std::string &dtype) const {

  auto op_pm_bin_map = op_pm_map_.op_to_pm_bin_map;
  std::string op = op_name; // + "_" + dtype;
  if (op_pm_bin_map.find(op) == op_pm_bin_map.end()) {
    throw std::runtime_error("Cannot find PM elf file for op: " + op_name +
                             " dtype: " + dtype);
  }

  return op_pm_bin_map.at(op);
}

const OpsFusion::OpPMMap::PMBinMetaInfo
pm_load::get_pmbin_meta(const std::string &pm_bin_name) const {

  auto pm_bin_meta_map = op_pm_map_.pm_bin_to_meta_map;
  if (pm_bin_meta_map.find(pm_bin_name) == pm_bin_meta_map.end()) {
    throw std::runtime_error("Cannot find PM bin meta for : " + pm_bin_name);
  }

  return pm_bin_meta_map.at(pm_bin_name);
}

const std::uint32_t
pm_load::get_pm_core_size(const std::string pm_name, const std::uint8_t c,
                          const std::uint8_t r,
                          const std::uint8_t num_cores) const {
  // TODO: index/size error checks
  auto pmbin_meta = get_pmbin_meta(pm_name);
  return pmbin_meta.pm_bin_core_offset[c * num_cores + r];
}

const std::uint32_t
pm_load::get_pm_core_offset(const std::string pm_name, const std::uint8_t c,
                            const std::uint8_t r,
                            const std::uint8_t num_cores) const {
  // TODO: index/size error checks
  auto pmbin_meta = get_pmbin_meta(pm_name);
  return pmbin_meta.pm_bin_core_size[c * num_cores + r];
}

void pm_load::execute(std::string op_name, std::string dtype) {

  const std::map<std::string, std::any> &attr{
      {"op_type", op_name}, {"op_dtype", dtype}, {"pm_id", (std::uint32_t)0}};

  std::vector<uint8_t> pm_bin = get_pm_bin(attr);
  std::cout << "pm_bin_file_size  " << pm_bin.size() << std::endl;

  auto pm_bo =
      xrt::bo(xrt_ctx_->get_device(), pm_bin.size(), XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  pm_bo.write(pm_bin.data());
  pm_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Prepare txn to load PM bin
  std::vector<Tensor> input, output;

  auto txn_bin = get_transaction_bin(input, output, attr);
  auto i_buf = transaction_op(txn_bin);
  size_t instr_bo_size = i_buf.get_txn_instr_size();
  auto instr_bo_ =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_size, xrt::bo::flags::cacheable,
              xrt_ctx_->get_kernel().group_id(1));
  instr_bo_.write(i_buf.get_txn_op().data());
  instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute kernel to load PM Bin
  auto kernel = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(kernel, 2, instr_bo_,
                                            instr_bo_.size() / sizeof(int),
                                            pm_bo, 0, 0, 0, 0, true, false);
}

const std::vector<uint8_t> pm_load::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  // find op_name
  std::string op_type;
  if (attr.find("op_type") != attr.end()) {
    op_type = std::any_cast<std::string>(attr.find("op_type")->second);
  } else {
    throw std::runtime_error("Can't find op_type in attrs");
  }

  std::string op_dtype;
  if (attr.find("op_dtype") != attr.end()) {
    op_dtype = std::any_cast<std::string>(attr.find("op_dtype")->second);
  } else {
    throw std::runtime_error("Can't find op_dtype in attrs");
  }

  uint32_t pm_id;
  if (attr.find("pm_id") != attr.end()) {
    pm_id = std::any_cast<std::uint32_t>(attr.find("pm_id")->second);
  } else {
    throw std::runtime_error("Can't find pm_id in attrs");
  }

  auto pmbin_name = get_op_pmbin_name(op_type, op_dtype);
  auto pmbin_meta = get_pmbin_meta(pmbin_name);

  // Initialize AIE Driver. Hardcode for STRIX for now
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  ConfigPtr.NumCols = overlay_meta_.num_cols;
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  XAie_LocType ShimDma;
  XAie_DmaDesc DmaDesc;
  patch_op_t patch_op;

  bool preemption = false;
  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  if (attr.end() != attr.find("preemption")) {
    preemption = std::any_cast<uint32_t>(attr.at("preemption"));
  }
  // preemption start marker.
  if (preemption) {
    XAie_Txn_PmLoadStart(&DevInst, pm_id);
  }
  // Reset Core tiles and Tile DMAs
  for (int c = 0; c < overlay_meta_.num_cols; c++) {
    for (int r = XAIE_AIE_TILE_ROW_START;
         r < XAIE_AIE_TILE_NUM_ROWS + XAIE_AIE_TILE_ROW_START; r++) {
      XAie_CoreDisable(&DevInst, XAie_TileLoc(c, r));
      XAie_CoreReset(&DevInst, XAie_TileLoc(c, r));
      XAie_CoreUnreset(&DevInst, XAie_TileLoc(c, r));
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r), DMA_CHANNEL_RESET);
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r),
                              DMA_CHANNEL_UNRESET);
    }
  }

  for (int c = 0; c < overlay_meta_.num_cols; c++) {
    for (int r = XAIE_MEM_TILE_ROW_START;
         r < XAIE_MEM_TILE_NUM_ROWS + XAIE_MEM_TILE_ROW_START; r++) {
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r), DMA_CHANNEL_RESET);
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r),
                              DMA_CHANNEL_UNRESET);
    }
  }
  auto &pkt_sw_meta = overlay_meta_.pkt_sw_meta;

  for (auto &meta_ : pkt_sw_meta) {
    // TODO: Handle packet id and come up with BD sequence for it.
    ShimDma = XAie_TileLoc(meta_.col, 0);

    for (int r = 0; r < meta_.num_cores; r++) {
      uint8_t BdId = r; // Always use BD 0 for PM configuration

      auto bin_offset =
          pmbin_meta.pm_bin_core_offset[meta_.col * meta_.num_cores +
                                        r]; // get_pm_core_size();
      auto bin_size = pmbin_meta.pm_bin_core_size[meta_.col * meta_.num_cores +
                                                  r]; // get_pm_core_offset();

      XAie_DmaDescInit(&DevInst, &DmaDesc, ShimDma);
      XAie_DmaSetAddrLen(&DmaDesc, 0, (uint32_t)bin_size);
      XAie_DmaEnableBd(&DmaDesc);
      XAie_DmaSetAxi(&DmaDesc, 0U, 32U, 0U, 2U, 0U);
      XAie_DmaWriteBd(&DevInst, &DmaDesc, ShimDma, BdId);
      // insert patch op for input
      patch_op.action = 0;
      patch_op.argidx = 0;
      patch_op.argplus = bin_offset;
      u64 regaddr = DmaDesc.DmaMod->BaseAddr +
                    BdId * DmaDesc.DmaMod->IdxOffset +
                    +_XAie_GetTileAddr(&DevInst, 0, meta_.col) +
                    DmaDesc.DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Idx * 4;
      patch_op.regaddr = regaddr;
      XAie_AddCustomTxnOp(&DevInst, XAIE_IO_CUSTOM_OP_DDR_PATCH,
                          (void *)&patch_op, sizeof(patch_op));

      XAie_DmaChannelSetStartQueue(&DevInst, ShimDma, meta_.dma_ch_num,
                                   DMA_MM2S, BdId, 1, XAIE_DISABLE);
    }
    XAie_DmaChannelEnable(&DevInst, ShimDma, meta_.dma_ch_num, DMA_MM2S);
  }

  // Poll for completition after all BD Writes are done
  for (auto &meta_ : pkt_sw_meta) {
    ShimDma = XAie_TileLoc(meta_.col, 0);
    XAie_DmaWaitForDone(&DevInst, ShimDma, 0, DMA_MM2S, 0);
  }

  // Reset all DMA Channels
  for (auto &meta_ : pkt_sw_meta) {
    ShimDma = XAie_TileLoc(meta_.col, 0);
    // TODO: Is this needed?
    // XAie_DmaChannelResetAll(&DevInst, ShimDma, DMA_CHANNEL_RESET);
    // XAie_DmaChannelResetAll(&DevInst, ShimDma, DMA_CHANNEL_UNRESET);
  }

  // Enable all cores
  if (op_type != "square" && op_type != "cube") {
    for (int c = 0; c < overlay_meta_.num_cols; c++) {
      for (int r = XAIE_AIE_TILE_ROW_START;
           r < XAIE_AIE_TILE_NUM_ROWS + XAIE_AIE_TILE_ROW_START; r++) {
        XAie_CoreEnable(&DevInst, XAie_TileLoc(c, r));
      }
    }
  }
  // preemption end marker.
  if (preemption) {
    XAie_Txn_PmLoadEnd(&DevInst);
  }

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

std::vector<OpArgMap>
pm_load::get_buffer_reqs(std::vector<Tensor> &input,
                         std::vector<Tensor> &output,
                         const std::map<std::string, std::any> &attr) const {

  auto pm_bin = get_pm_bin(attr);
  // Load PM in super_kernel_param_input
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 0, 0, pm_bin.size()}};
  return arg_map;
}

const std::vector<uint8_t> pm_load::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  return get_pm_bin(attr);
}

const std::vector<uint8_t>
pm_load::get_pm_bin(const std::map<std::string, std::any> &attr) const {
  // find op_name
  std::string op_type;
  if (attr.find("op_type") != attr.end()) {
    op_type = std::any_cast<std::string>(attr.find("op_type")->second);
  } else {
    throw std::runtime_error("Can't find op_type in attrs");
  }

  std::string op_dtype;
  if (attr.find("op_dtype") != attr.end()) {
    op_dtype = std::any_cast<std::string>(attr.find("op_dtype")->second);
  } else {
    throw std::runtime_error("Can't find op_dtype in attrs");
  }

  auto pm_file_name = get_op_pmbin_name(op_type, op_dtype);
  auto &pm_container = Preemption::getInstance();
  std::string pm_key =
      PM_PREFIX + pm_file_name.substr(0, pm_file_name.length() - 4);
  auto pm_bin = pm_container.get_pm_bvec(pm_key);
  return pm_bin;
}

} // namespace ryzenai
