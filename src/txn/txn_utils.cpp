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

#include "utils/ipu_hw_config.hpp"
#include "utils/op_utils.hpp"
#include <utils/tfuncs.hpp>

#include "txn_utils.hpp"

namespace utils {

static constexpr size_t SUPER_KERNEL_ARGIDX = 4;

void txn_util::pass_through(uint8_t **ptr) {
  auto op_hdr = (XAie_OpHdr *)(*ptr);
  switch (op_hdr->Op) {
  case XAIE_IO_WRITE: {
    XAie_Write32Hdr *w_hdr = (XAie_Write32Hdr *)(*ptr);
    *ptr = *ptr + w_hdr->Size;
    break;
  }
  case XAIE_IO_BLOCKWRITE: {
    XAie_BlockWrite32Hdr *bw_header = (XAie_BlockWrite32Hdr *)(*ptr);
    *ptr = *ptr + bw_header->Size;
    break;
  }
  case XAIE_IO_MASKWRITE: {
    XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)(*ptr);
    *ptr = *ptr + mw_header->Size;
    break;
  }
  case XAIE_IO_MASKPOLL:
  case XAIE_IO_MASKPOLL_BUSY: {
    XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)(*ptr);
    *ptr = *ptr + mp_header->Size;
    break;
  }
  case (XAIE_IO_CUSTOM_OP_TCT):
  case (XAIE_IO_CUSTOM_OP_DDR_PATCH):
  case (XAIE_IO_CUSTOM_OP_READ_REGS):
  case (XAIE_IO_CUSTOM_OP_RECORD_TIMER):
  case (XAIE_IO_CUSTOM_OP_MERGE_SYNC): {
    XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
    *ptr = *ptr + Hdr->Size;
    break;
  }
  case (XAIE_IO_PREEMPT): {
    XAie_PreemptHdr *Hdr = (XAie_PreemptHdr *)(*ptr);
    *ptr = *ptr + sizeof(*Hdr);
    break;
  }
  default:
    // RYZENAI_LOG_TRACE("Opcode: " + std::to_string(op_hdr->Op));
    throw std::runtime_error("Unknown op to pass through");
  }
}

/**
 * patch the txn so that operators that have merged input can keep their inputs
 * in seperated buffers based on the destination specification
 *
 * @param txn base transaction for bo splitting
 * @param source_args_map original buffer req with inputs in the same bo
 * (doesn't need to be complete)
 * @param dest_args_map desired buffer configuration (doesn't need to be
 * complete)
 *
 * @returns modified txn that has args based on the desired buffer configuration
 */
std::vector<uint8_t>
txn_util::patch(const std::vector<uint8_t> &txn,
                const std::vector<OpArgMap> &source_args_map,
                const std::vector<OpArgMap> &dest_args_map) {
  auto txn_patch = txn;
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_patch.data();
  int num_ops = Hdr->NumOps;
  uint8_t *ptr = txn_patch.data() + sizeof(*Hdr);
  const auto argmap_partition =
      dynamic_dispatch::op_utils::partition_argmap(source_args_map);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Total #ops : {}", num_ops));
  DD_THROW_IF(
      dest_args_map.size() != source_args_map.size(),
      OpsFusion::dd_format("dest_args_map size({}) != args_map size({})",
                           dest_args_map.size(), source_args_map.size()))
  for (int n = 0; n < num_ops; n++) {
    auto op_hdr = (XAie_OpHdr *)ptr;
    switch (op_hdr->Op) {
    case XAIE_IO_CUSTOM_OP_DDR_PATCH: {
      XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(ptr);
      u32 size = hdr->Size;
      patch_op_t *op = (patch_op_t *)((ptr) + sizeof(*hdr));

      const auto curr_argidx = op->argidx;
      const auto curr_offset = op->argplus;
      // don't need to specify the full arg map, just to the max xrt idx where
      // there are changes
      if (curr_argidx < argmap_partition.size()) {
        const OpArgMap &op_arg = dynamic_dispatch::op_utils::find_op_arg(
            argmap_partition, curr_argidx, curr_offset);
        // look up the argmap in the source ar map and change it to the dest
        // argmap
        for (size_t i = 0; i < source_args_map.size(); i++) {
          if ((op_arg.arg_type == source_args_map.at(i).arg_type) &&
              (op_arg.xrt_arg_idx == source_args_map.at(i).xrt_arg_idx) &&
              (op_arg.onnx_arg_idx == source_args_map.at(i).onnx_arg_idx) &&
              (op_arg.offset == source_args_map.at(i).offset)) {
            op->argplus = dest_args_map.at(i).offset + curr_offset -
                          source_args_map.at(i).offset;
            op->argidx = dest_args_map.at(i).xrt_arg_idx;
            RYZENAI_LOG_TRACE(OpsFusion::dd_format(
                "Patched : [{}:{}] -> [ ARG TYPE: {} ] -> [{}:{}] ",
                curr_argidx, curr_offset, source_args_map.at(i).xrt_arg_idx,
                op->argidx, op->argplus));
            break;
          }
        }
      }
      ptr = ptr + size;
    } break;
    default:
      // no modification for other ops
      pass_through(&ptr);
      break;
    }
  }
  return txn_patch;
}

void txn_util::append_to_txn(XAie_DevInst *DevInst, uint8_t **ptr) {
  auto op_hdr = (XAie_OpHdr *)(*ptr);
  switch (op_hdr->Op) {
  case XAIE_IO_WRITE: {
    XAie_Write32Hdr *w_hdr = (XAie_Write32Hdr *)(*ptr);
    XAie_Write32(DevInst, w_hdr->RegOff, w_hdr->Value);
    *ptr = *ptr + w_hdr->Size;
    break;
  }
  case XAIE_IO_BLOCKWRITE: {
    XAie_BlockWrite32Hdr *bw_header = (XAie_BlockWrite32Hdr *)(*ptr);
    u32 size = (bw_header->Size - sizeof(*bw_header)) / 4;
    u32 *payload = (u32 *)((*ptr) + sizeof(*bw_header));
    XAie_BlockWrite32(DevInst, bw_header->RegOff, payload, size);
    *ptr = *ptr + bw_header->Size;
    break;
  }
  case XAIE_IO_MASKWRITE: {
    XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)(*ptr);
    XAie_MaskWrite32(DevInst, mw_header->RegOff, mw_header->Mask,
                     mw_header->Value);
    *ptr = *ptr + mw_header->Size;
    break;
  }
  case XAIE_IO_MASKPOLL: {
    XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)(*ptr);
    XAie_MaskPoll(DevInst, mp_header->RegOff, mp_header->Mask, mp_header->Value,
                  0);
    *ptr = *ptr + mp_header->Size;
    break;
  }
  case (XAIE_IO_PREEMPT): {
    XAie_PreemptHdr *hdr = (XAie_PreemptHdr *)(*ptr);
    XAie_Txn_Preempt(DevInst, hdr);
    *ptr = *ptr + sizeof(*hdr);
    break;
  }
  case (XAIE_IO_CUSTOM_OP_TCT): {
    XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
    tct_op_t *op = (tct_op_t *)((*ptr) + sizeof(*hdr));
    XAie_AddCustomTxnOp(DevInst, XAIE_IO_CUSTOM_OP_TCT, (void *)op,
                        sizeof(*op));
    *ptr = *ptr + hdr->Size;
    break;
  }
  case (XAIE_IO_CUSTOM_OP_DDR_PATCH): {
    XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
    patch_op_t *op = (patch_op_t *)((*ptr) + sizeof(*hdr));
    XAie_AddCustomTxnOp(DevInst, XAIE_IO_CUSTOM_OP_DDR_PATCH, (void *)op,
                        sizeof(*op));
    *ptr = *ptr + hdr->Size;
    break;
  }
  case (XAIE_IO_CUSTOM_OP_READ_REGS): {
    XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
    read_register_op_t *op = (read_register_op_t *)((*ptr) + sizeof(*hdr));
    XAie_AddCustomTxnOp(DevInst, XAIE_IO_CUSTOM_OP_READ_REGS, (void *)op,
                        sizeof(*op));
    *ptr = *ptr + hdr->Size;
    break;
  }
  case (XAIE_IO_CUSTOM_OP_RECORD_TIMER): {
    XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
    record_timer_op_t *op = (record_timer_op_t *)((*ptr) + sizeof(*hdr));
    XAie_AddCustomTxnOp(DevInst, XAIE_IO_CUSTOM_OP_RECORD_TIMER, (void *)op,
                        sizeof(*op));
    *ptr = *ptr + hdr->Size;
    break;
  }
  case (XAIE_IO_CUSTOM_OP_MERGE_SYNC): {
    XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
    tct_op_t *op = (tct_op_t *)((*ptr) + sizeof(*hdr));
    XAie_AddCustomTxnOp(DevInst, XAIE_IO_CUSTOM_OP_MERGE_SYNC, (void *)op,
                        sizeof(*op));
    *ptr = *ptr + hdr->Size;
    break;
  }
  default:
    // RYZENAI_LOG_TRACE("Opcode: " + std::to_string(op_hdr->Op));
    throw std::runtime_error("Unknown op to pass through");
  }
}

std::vector<uint8_t>
txn_util::convert_to_opt_txn(const std::vector<uint8_t> &base_txn) {

  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)base_txn.data();
  // Initialize AIE Driver. Hardcode for STRIX for now
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  ConfigPtr.NumCols = Hdr->NumCols;
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "Before converting v1 to v2, transaction size, num_ops: {}, {}",
      Hdr->TxnSize, Hdr->NumOps));
  int num_ops = Hdr->NumOps;
  uint8_t *ptr = (uint8_t *)base_txn.data() + sizeof(*Hdr);
  for (int n = 0; n < num_ops; n++) {
    append_to_txn(&DevInst, &ptr);
  }

  uint8_t *txn_ptr = XAie_ExportSerializedTransaction_opt(&DevInst, 0, 0);
  Hdr = (XAie_TxnHeader *)txn_ptr;
  auto size = Hdr->TxnSize;
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "After converting v1 to v2, transaction size, num_ops: {}, {}",
      Hdr->TxnSize, Hdr->NumOps));

  std::vector<uint8_t> txn(size, 0);
  memcpy((void *)txn.data(), (void *)txn_ptr, size);

  // check if there is an API to free txn pointer
  free(txn_ptr);
  XAie_Finish(&DevInst);
  return txn;
}

/**
 * overloaded patch for tiling of kernels, will be executed during runtime
 * dynamic shape fusion, patch different operations based on the offset map
 *
 * @param txn: base transaction bin for tiling
 * @param args_map: Using the offset attribute in the args map for patching
 *
 * @returns patched transaction code based on the offset value.
 */
std::vector<uint8_t> txn_util::patch(const std::vector<uint8_t> &txn,
                                     const std::vector<OpArgMap> &args_map) {

  auto txn_patch = txn;
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_patch.data();
  int num_ops = Hdr->NumOps;
  uint8_t *ptr = txn_patch.data() + sizeof(*Hdr);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Total #ops : {}", num_ops));

  for (int n = 0; n < num_ops; n++) {
    auto op_hdr = (XAie_OpHdr *)ptr;
    switch (op_hdr->Op) {
    case XAIE_IO_CUSTOM_OP_DDR_PATCH: {
      XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(ptr);
      u32 size = hdr->Size;
      patch_op_t *op = (patch_op_t *)((ptr) + sizeof(*hdr));

      const auto curr_argidx = op->argidx;
      const auto curr_offset = op->argplus;
      for (auto arg : args_map) {
        if (arg.xrt_arg_idx == curr_argidx) {
          op->argplus = arg.offset + curr_offset;
          RYZENAI_LOG_TRACE(OpsFusion::dd_format(
              "Patched : [{}:{}] -> [ ARG TYPE: {} ] -> [{}:{}] ", curr_argidx,
              curr_offset, arg.xrt_arg_idx, op->argidx, op->argplus));
          break;
        }
      }

      ptr = ptr + size;
    } break;
    default:
      // no modification for other ops
      pass_through(&ptr);
      break;
    }
  }

  return txn_patch;
}

void txn_util::patch(const OpsFusion::Metadata::OpInfo &op_info,
                     const OpsFusion::Metadata &meta,
                     const std::vector<OpArgMap> &args_map) {
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Patching Instructions for op:{} ...",
                                         op_info.name));
  const auto &tensor_map = meta.tensor_map;
  const auto &super_instr_map = meta.super_instr_map;
  const auto &const_map = meta.const_map;
  const auto intermediate_scratch_size =
      MAP_AT(meta.fused_tensors, "scratch").size;
  const auto max_op_scratch_pad_size = meta.max_op_scratch_pad_size;
  auto args = OpsFusion::get_op_args(op_info);
  auto total_args_size = args.size();

  const auto argmap_partition =
      dynamic_dispatch::op_utils::partition_argmap(args_map);

  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn.data();
  int num_ops = Hdr->NumOps;
  uint8_t *ptr = txn.data() + sizeof(*Hdr);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Total #ops : {}, max_op_scratch_pad_size: {}",
                           num_ops, max_op_scratch_pad_size));

  for (int n = 0; n < num_ops; n++) {
    auto op_hdr = (XAie_OpHdr *)ptr;
    switch (op_hdr->Op) {
    case XAIE_IO_CUSTOM_OP_DDR_PATCH: {
      XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(ptr);
      std::uint32_t size = hdr->Size;
      patch_op_t *op = (patch_op_t *)((ptr) + sizeof(*hdr));

      const auto curr_argidx = op->argidx;
      const auto curr_offset = op->argplus;

      // support two additional args for super kernels and initlize const params
      // super kernel params can be sent to NPU - ONNX node may not have this as
      // an input to the op some operators may need to send LUTs to NPU from DDR
      // for functionality. This will not be represented as an input in onnx
      // node. Example kernels - bf16 Silu/Gelu ops.
      DD_THROW_IF((curr_argidx >= total_args_size + 2),
                  OpsFusion::dd_format("curr_argidx({}) >= # op_args({}) + 2",
                                       curr_argidx, total_args_size));
      DD_THROW_IF(curr_argidx >= args_map.size(),
                  OpsFusion::dd_format("curr_argidx({}) >= args_map size({})",
                                       curr_argidx, args_map.size()));

      const auto &op_arg = dynamic_dispatch::op_utils::find_op_arg(
          argmap_partition, curr_argidx, curr_offset);

      if (op_arg.arg_type == OpArgMap::CONST_KERNEL_PARAM_INPUT) {
        op->argidx = OpArgMap::CONST_KERNEL_PARAM_INPUT;
        op->argplus = curr_offset + super_instr_map.at(op_info.name).offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patched : [{}:{}] -> [ super kernel instr ] -> [{}:{}] ",
            curr_argidx, curr_offset, op->argidx, op->argplus));
      } else if (op_arg.arg_type == OpArgMap::CONST_INPUT) {
        op->argidx = OpArgMap::CONST_INPUT;
        op->argplus = curr_offset + MAP_AT(const_map, op_info.name).offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patched : [{}:{}] -> [ const ] -> [{}:{}] ", curr_argidx,
            curr_offset, op->argidx, op->argplus));
      } else if (op_arg.arg_type == OpArgMap::CTRL_PKT_BIN) {
        // Ctrl Pkt bin will be packed with super kernel instructions BO
        const auto &aux_info = meta.aux_info;
        if (aux_info.find("elf_flow") != aux_info.end() &&
            std::any_cast<bool>(aux_info.at("elf_flow"))) {
          op->argidx = OpArgMap::CTRL_PKT_BIN;
          op->argplus = curr_offset + meta.ctrl_pkt_map.at(op_info.name).offset;
        } else {
          auto super_kernel_size = meta.fused_tensors.at("super_instr").size;
          op->argidx = OpArgMap::CONST_KERNEL_PARAM_INPUT;
          op->argplus = curr_offset +
                        meta.ctrl_pkt_map.at(op_info.name).offset +
                        super_kernel_size;
        }

        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patched : [{}:{}] -> [ ctrl pkt bin ] -> [{}:{}] ", curr_argidx,
            curr_offset, op->argidx, op->argplus));
      } else if (op_arg.arg_type == OpArgMap::SCRATCH_PAD) {
        op->argidx = OpArgMap::SCRATCH_PAD;
        DD_THROW_IF(
            curr_offset >= max_op_scratch_pad_size,
            OpsFusion::dd_format(
                "curr_offset({}) >= args_map max_op_scratch_pad_size({})",
                curr_offset, max_op_scratch_pad_size));
        // Note: Internal scratch pad for each op is shared, since it
        //       is assumed ops will execute sequentially
        //       Offset by scratch buffer size since beginning will store
        //       intermediate outputs, i.e. memory layout will be
        //       [intermediate_outputs | internal_scratch_pad]
        op->argplus = curr_offset + intermediate_scratch_size;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patched : [{}:{}] -> [ scratch pad mem ] -> [{}:{}] ", curr_argidx,
            curr_offset, op->argidx, op->argplus));
      } else {
        const size_t onnx_argidx = op_arg.onnx_arg_idx;
        const auto &arg_label = ARRAY_AT(args, onnx_argidx);
        const auto &tensor = MAP_AT(tensor_map, arg_label);

        size_t new_argidx = tensor.arg_idx;
        size_t block_offset = tensor.offset;
        size_t curr_offset_delta = curr_offset - op_arg.offset;
        // tensor.offset tells where data actually is
        // op_arg.padding_offset is op requirement on whether it needs address
        // of actual data or beginning of padding
        size_t final_offset =
            block_offset + curr_offset_delta - op_arg.padding_offset;

        op->argidx = new_argidx;
        op->argplus = final_offset;
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patched : [{}:{}] -> [ onnx_argid:{} ] -> [{}:{}] ", curr_argidx,
            curr_offset, onnx_argidx, op->argidx, op->argplus));
      }

      ptr = ptr + size;

    } break;
    default:
      // no modification for other ops
      pass_through(&ptr);
      break;
    }
  }
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "Patching Instructions for op:{} ... DONE", op_info.name));
}

std::vector<uint8_t> txn_util::to_vector() { return txn; }

std::string txn_util::summarize() {
  std::stringstream ss;
  ss << ss_hdr_.str() << ss_summary_.str();

  return ss.str();
}

std::string txn_util::text_dump() {
  std::stringstream ss;
  ss << ss_hdr_.str() << ss_ops_.str();
  return ss.str();
}

txn_util::txn_util(const std::vector<uint8_t> &txn_vec) {
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_vec.data();
  if (txn_vec.size() != Hdr->TxnSize) {
    throw std::runtime_error(
        "Invalid Transaction Vec : Size of input transaction vector and the "
        "size specified in its header doesn't match.");
  }

  txn.resize(Hdr->TxnSize);
  std::memcpy(txn.data(), txn_vec.data(), Hdr->TxnSize);
  txn_size_ = Hdr->TxnSize;
  num_txn_ops_ = Hdr->NumOps;
}

void txn_util::prepare_summary() {
  ss_summary_ << "Summary of transaction binary" << std::endl;
  ss_summary_ << "Number of write ops: " << std::to_string(num_w_ops)
              << std::endl;
  ss_summary_ << "Number of block_write ops: " << std::to_string(num_bw_ops)
              << std::endl;
  ss_summary_ << "Number of mask_write ops: " << std::to_string(num_mw_ops)
              << std::endl;
  ss_summary_ << "Number of mask_poll ops: " << std::to_string(num_mp_ops)
              << std::endl;
  ss_summary_ << "Number of tct ops: " << std::to_string(num_tct_ops)
              << std::endl;
  ss_summary_ << "Number of patch ops: " << std::to_string(num_patch_ops)
              << std::endl;
  ss_summary_ << "Number of read ops: " << std::to_string(num_read_ops)
              << std::endl;
  ss_summary_ << "Number of timer ops: " << std::to_string(num_readtimer_ops)
              << std::endl;
  ss_summary_ << "Number of merge sync ops: "
              << std::to_string(num_mergesync_ops) << std::endl;
}

void txn_util::stringify_w32(uint8_t **ptr) {
  XAie_Write32Hdr *w_hdr = (XAie_Write32Hdr *)(*ptr);
  ss_ops_ << "W: 0x" << std::hex << w_hdr->RegOff << " 0x" << w_hdr->Value
          << std::endl;
  *ptr = *ptr + w_hdr->Size;
  num_w_ops++;
}

void txn_util::stringify_bw32(uint8_t **ptr) {
  XAie_BlockWrite32Hdr *bw_header = (XAie_BlockWrite32Hdr *)(*ptr);
  std::uint32_t bw_size = bw_header->Size;
  std::uint32_t Size = (bw_size - sizeof(*bw_header)) / 4;
  std::uint32_t *Payload = (std::uint32_t *)((*ptr) + sizeof(*bw_header));
  ss_ops_ << "BW: 0x" << std::hex << bw_header->RegOff << " ";
  // ss_ops_ << "Payload: ";
  for (std::uint32_t i = 0; i < Size; i++) {
    ss_ops_ << "0x" << std::hex << *Payload << " ";
    Payload++;
  }
  ss_ops_ << std::endl;
  *ptr = *ptr + bw_size;
  num_bw_ops++;
}

void txn_util::stringify_mw32(uint8_t **ptr) {
  XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)(*ptr);
  ss_ops_ << "MW: 0x" << std::hex << mw_header->RegOff << " " << mw_header->Mask
          << " " << mw_header->Value << std::endl;
  *ptr = *ptr + mw_header->Size;
  num_mw_ops++;
}

void txn_util::stringify_mp32(uint8_t **ptr) {
  XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)(*ptr);
  ss_ops_ << "MP: 0x" << std::hex << mp_header->RegOff << " " << mp_header->Mask
          << " " << mp_header->Value << std::endl;
  *ptr = *ptr + mp_header->Size;
  num_mp_ops++;
}

void txn_util::stringify_tct(uint8_t **ptr) {
  XAie_CustomOpHdr *co_header = (XAie_CustomOpHdr *)(*ptr);
  ss_ops_ << "TCT: " << std::endl;
  *ptr = *ptr + co_header->Size;
  num_tct_ops++;
}

void txn_util::stringify_patchop(uint8_t **ptr) {
  XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
  std::uint32_t size = hdr->Size;
  ss_ops_ << "PatchOp: ";
  patch_op_t *op = (patch_op_t *)((*ptr) + sizeof(*hdr));
  auto reg_off = op->regaddr;
  auto arg_idx = op->argidx;
  auto addr_offset = op->argplus;
  ss_ops_ << "(RegAddr: " << std::hex << reg_off << " Arg Idx: " << arg_idx
          << " Addr Offset: " << addr_offset << ")" << std::endl;
  *ptr = *ptr + size;
  num_patch_ops++;
}

void txn_util::stringify_rdreg(uint8_t **ptr) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  std::uint32_t size = Hdr->Size;
  ss_ops_ << "ReadOp: " << std::endl;
  *ptr = *ptr + size;
  num_read_ops++;
}

void txn_util::stringify_rectimer(uint8_t **ptr) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  std::uint32_t size = Hdr->Size;
  ss_ops_ << "TimerOp: " << std::endl;
  *ptr = *ptr + size;
  num_readtimer_ops++;
}

void txn_util::stringify_mergesync(uint8_t **ptr) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  std::uint32_t size = Hdr->Size;
  ss_ops_ << "MergeSyncOp: " << std::endl;
  *ptr = *ptr + size;
  num_mergesync_ops++;
}

void txn_util::stringify_txn_ops() {
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn.data();
  auto num_ops = Hdr->NumOps;
  auto ptr = txn.data() + sizeof(*Hdr);

  XAie_OpHdr *op_hdr;
  for (uint32_t i = 0; i < num_ops; i++) {
    op_hdr = (XAie_OpHdr *)ptr;
    // printf("i: %d, OpCode: %d\n", i, op_hdr->Op);
    switch (op_hdr->Op) {
    case XAIE_IO_WRITE:
      stringify_w32(&ptr);
      break;
    case XAIE_IO_BLOCKWRITE:
      stringify_bw32(&ptr);
      break;
    case XAIE_IO_MASKWRITE:
      stringify_mw32(&ptr);
      break;
    case XAIE_IO_MASKPOLL:
      stringify_mp32(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_TCT:
      stringify_tct(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_DDR_PATCH:
      stringify_patchop(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_READ_REGS:
      stringify_rdreg(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_RECORD_TIMER:
      stringify_rectimer(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_MERGE_SYNC:
      stringify_mergesync(&ptr);
      break;
    default:
      throw std::runtime_error("Error: Unknown op code at offset at " +
                               std::to_string(ptr - txn.data()) +
                               ". OpCode: " + std::to_string(op_hdr->Op));
    }
  }
}

void txn_util::stringify_txn_bin() {

  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn.data();

  ss_hdr_ << "Header version: " << std::to_string(Hdr->Major) << "."
          << std::to_string(Hdr->Minor) << std::endl;
  ss_hdr_ << "Device Generation: " << std::to_string(Hdr->DevGen) << std::endl;
  ss_hdr_ << "Partition Info: " << std::endl;
  ss_hdr_ << "     Num Cols:" << std::to_string(Hdr->NumCols) << std::endl;
  ss_hdr_ << "     Num Rows:" << std::to_string(Hdr->NumRows) << std::endl;
  ss_hdr_ << "     Num MemTile Rows:" << std::to_string(Hdr->NumMemTileRows)
          << std::endl;
  ss_hdr_ << "Transaction Metadata:" << std::endl;
  ss_hdr_ << "     Size: " << std::to_string(Hdr->TxnSize) << std::endl;
  ss_hdr_ << "     NumOps: " << std::to_string(Hdr->NumOps) << std::endl;

  stringify_txn_ops();
}

std::vector<uint8_t>
txn_util::fuse_txns(const std::vector<std::vector<uint8_t>> &txns) {
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Fuse {} transactions ...", txns.size()));
  DD_ASSERT(!txns.empty(), "No transactions to fuse");

  std::vector<uint8_t> fused_txn;

  uint32_t NumOps = uint32_t(0);
  uint32_t TxnSize = uint32_t(sizeof(XAie_TxnHeader));

  // First go through all txn and figure out size to pre-allocate
  // this is to avoid unnecessary vector re-allocation
  for (size_t i = 0; i < txns.size(); ++i) {
    const auto &txn = ARRAY_AT(txns, i);
    const XAie_TxnHeader *txn_hdr = (const XAie_TxnHeader *)txn.data();
    NumOps += txn_hdr->NumOps;

    DD_ASSERT(txn_hdr->TxnSize > sizeof(XAie_TxnHeader),
              OpsFusion::dd_format(
                  "Size of fused_transaction {} smaller than its header {}",
                  txn_hdr->TxnSize, sizeof(XAie_TxnHeader)));
    uint32_t instr_size = txn_hdr->TxnSize - uint32_t(sizeof(XAie_TxnHeader));
    TxnSize += instr_size;
  }

  fused_txn.reserve(TxnSize);

  // First txn - copy over header too
  const auto &txn1 = ARRAY_AT(txns, 0);
  const XAie_TxnHeader *txn1_hdr = (const XAie_TxnHeader *)txn1.data();
  fused_txn.insert(fused_txn.end(), txn1.data(),
                   txn1.data() + txn1_hdr->TxnSize);

  // Rest of txns
  for (size_t i = 1; i < txns.size(); ++i) {
    const auto &txn = ARRAY_AT(txns, i);
    const XAie_TxnHeader *txn_hdr = (const XAie_TxnHeader *)txn.data();
    const uint8_t *instr_ptr = txn.data() + sizeof(XAie_TxnHeader);
    // skip copying over the header for the rest of txns
    size_t instr_size = txn_hdr->TxnSize - sizeof(XAie_TxnHeader);
    fused_txn.insert(fused_txn.end(), instr_ptr, instr_ptr + instr_size);
  }

  // Update the header
  XAie_TxnHeader *txn_vec_hdr = (XAie_TxnHeader *)(fused_txn.data());
  txn_vec_hdr->NumOps = NumOps;
  txn_vec_hdr->TxnSize = TxnSize;
  DD_ASSERT(fused_txn.size() == TxnSize,
            OpsFusion::dd_format(
                "Size of fused_transaction {} doesn't match the size "
                "in its header {}",
                fused_txn.size(), TxnSize));

  // Just print summary.
  txn_util res(fused_txn);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Fused Ops Summary\n{}", res.summarize()));
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Fuse {} transactions ... DONE", txns.size()));
  return fused_txn;
}

#define OP_NOOP (0)
#define OP_WRITEBD 1
#define OP_WRITE32 2
#define OP_SYNC 3
#define OP_WRITEBD_EXTEND_AIETILE 4
#define OP_WRITE32_EXTEND_GENERAL 5
#define OP_WRITEBD_EXTEND_SHIMTILE 6
#define OP_WRITEBD_EXTEND_MEMTILE 7
#define OP_WRITE32_EXTEND_DIFFBD 8
#define OP_WRITEBD_EXTEND_SAMEBD_MEMTILE 9
#define OP_DUMPDDR 10
#define OP_WRITESHIMBD 11
#define OP_WRITEMEMBD 12
#define OP_WRITE32_RTP 13

// HW VERIFICATION OPS
#define OP_READ32 (14)
#define OP_READ32_POLL (15)

// Custom Record Timestamp OP
#define OP_RECORD_TIMESTAMP (16)

// new sync op that only looks at count of TCT
#define OP_MERGESYNC (17)

// size of payload of 32-bit words
#define OP_READ32_INCR (2)
#define OP_READ32_POLL_INCR (4)
#define OP_MERGESYNC_INCR (1)

#define ERR_SUCCESS (0)
#define ERR_UNSUPPORTED_FW_OP_CODE (1 << 0)
#define ERR_UNKNOWN_FW_OP_CODE (1 << 1)
#define ERR_UNKNOWN_INSTR_OP_CODE (1 << 2)
#define ERR_UNKNOWN_BUFFER_TYPE (1 << 3)

#define ERR_READ_FAILED (1 << 30)

#define GetInstrOpcode(word) ((((word) & 0xFF000000)) >> 24)
#define GetInstrTileCol(word) ((((word) & 0x00FF0000)) >> 16)
#define GetInstrTileRow(word) ((((word) & 0x0000FF00)) >> 8)

typedef uint32_t uint;

static u64 PHYSIC_DDR_ADDR_ifm_start;
static u64 PHYSIC_DDR_ADDR_param_start;
static u64 PHYSIC_DDR_ADDR_ofm_start;
static u64 PHYSIC_DDR_ADDR_inter_start;
static u64 PHYSIC_DDR_ADDR_out_2;
static u64 PHYSIC_DDR_ADDR_mc_code_start;
static u64 PHYSIC_DDR_ADDR_instruction_start;

static bool PATCHED = false; // Whether instruction buffer is patched by host
static bool PREPOST = false;
#define PP_ARGS_MAX_N 8
static u64 PHYSIC_DDR_ADDR_PP[PP_ARGS_MAX_N];

#define IFM_TYPE 0x0
#define PARAM_TYPE 0x1
#define OFM_TYPE 0x2
#define INTER_TYPE 0x3
#define OUT2_TYPE 0x4
#define MC_CODE_TYPE 0X5

#define DDR_ADDR_SHIFT 48
#define DDR_TYPE_MASK 0x7
#define DDR_ADDR_MASK 0xffffffffffff

#define SHIM_DMA_BD0_BASE_ADDR 0x1D000
#define SHIM_BD_OFFSET 0x20
#define MEM_DMA_BD0_BASE_ADDR 0xA0000
#define MEM_BD_OFFSET 0x20
#define AIE_DMA_BD0_BASE_ADDR 0x1D000
#define AIE_BD_OFFSET 0x20

u64 MapDDRAddrFromLogicToPhysicPP(u64 addr, std::uint8_t type_mask,
                                  int32_t *const status_p) {
  if (type_mask >= PP_ARGS_MAX_N) {
    // printf("error: ddr range is not known, type_mask = 0x%x.\n", type_mask);
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "error: ddr range is not known, type_mask = {}", type_mask));
    *status_p |= ERR_UNKNOWN_BUFFER_TYPE;
#ifdef __DPU_FW_ENABLE_ABORT__
    abort();
#endif
  } else {
    addr += PHYSIC_DDR_ADDR_PP[type_mask];
  }
  return addr;
}

u64 MapDDRAddrFromLogicToPhysic(u64 addr, std::uint8_t type_mask,
                                int32_t *const status_p) {
  if (PATCHED) {
    return addr;
  }

  if (PREPOST) {
    return MapDDRAddrFromLogicToPhysicPP(addr, type_mask, status_p);
  }

  if (type_mask == IFM_TYPE) {
    addr += PHYSIC_DDR_ADDR_ifm_start;
  } else if (type_mask == PARAM_TYPE) {
    addr += PHYSIC_DDR_ADDR_param_start;
  } else if (type_mask == OFM_TYPE) {
    addr += PHYSIC_DDR_ADDR_ofm_start;
  } else if (type_mask == INTER_TYPE) {
    addr += PHYSIC_DDR_ADDR_inter_start;
  } else if (type_mask == OUT2_TYPE) {
    addr += PHYSIC_DDR_ADDR_out_2;
  } else if (type_mask == MC_CODE_TYPE) {
    addr += PHYSIC_DDR_ADDR_mc_code_start;
  } else {
    // printf("error: ddr range is not known, type_mask = 0x%x.\n", type_mask);
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "error: ddr range is not known, type_mask = {}", type_mask));
    *status_p |= ERR_UNKNOWN_BUFFER_TYPE;
#ifdef __DPU_FW_ENABLE_ABORT__
    abort();
#endif
  }
  return addr;
}

// #define MC_CONVERSION_DEBUG

int32_t ExecNoOp(size_t *const pc) {
  // const std::uint32_t *iptr = instr_ptr + (*pc);
  // std::uint8_t Col = GetInstrTileCol(word);
  // std::uint8_t Row = GetInstrTileRow(word);

  int32_t status = ERR_SUCCESS;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info] ExecNoOp over, pc = %d\n", *pc);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info] ExecNoOp over, pc = k = {}", *pc));
#endif

  *pc += 1; // payload should just be 1 32-bit word

  return status;
}

int32_t ExecWriteBdOpt(const std::uint32_t *instr_ptr, size_t *pc,
                       u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t BdId = 0;
  std::uint8_t DDRType = 0;

  int32_t status = ERR_SUCCESS;

  if (Row == 0) { // shim-tile need update addr
    BdId = ((word) & 0x0000000F);
    DDRType = ((word) & 0x000000F0) >> 4;
    u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                   (Row << XAIE_ROW_SHIFT) + SHIM_DMA_BD0_BASE_ADDR +
                   (BdId * SHIM_BD_OFFSET);
    u64 AddrLow = iptr[2];
    u64 AddrHigh = (iptr[3] & 0x0000FFFF);
    u64 TensorAddr = ((((u64)AddrHigh) << 32) | AddrLow);
    TensorAddr = MapDDRAddrFromLogicToPhysic(TensorAddr, DDRType, &status);
    // printf("[info]: pc = 0x%08x, %08x, ExecWriteShimBd: {col, row, bd, "
    //        "ddr_type, ddr_addr} = {%d, %d, %d, %d, %d}\n",
    //        *pc, word, Col, Row, BdId, DDRType, TensorAddr);
    if (ERR_SUCCESS != status) {
      return status;
    }
    std::uint32_t tWord0 =
        std::uint32_t((TensorAddr) & 0xFFFFFFFFC); // unused 2-LSB
    std::uint32_t tWord1 =
        std::uint32_t((iptr[3] & 0xFFFF0000) | (TensorAddr >> 32));
#if 0
    opt_write32(BaseAddr, iptr[1]);
    opt_write32(BaseAddr+4u, tWord0);
    opt_write32(BaseAddr+8u, tWord1);
    opt_write32(BaseAddr+12u, iptr[4]);
    opt_write32(BaseAddr+16u, iptr[5]);
    opt_write32(BaseAddr+20u, iptr[6]);
    opt_write32(BaseAddr+24u, iptr[7]);
    opt_write32(BaseAddr+28u, iptr[8]);
#elif 0
    std::uint32_t *bd = (std::uint32_t *)BaseAddr;
    *(bd++) = iptr[1];
    *(bd++) = tWord0;
    *(bd++) = tWord1;
    *(bd++) = iptr[4];
    *(bd++) = iptr[5];
    *(bd++) = iptr[6];
    *(bd++) = iptr[7];
    *(bd++) = iptr[8];
#endif
    *pc += 9;
    // printf("[info] ExecWriteShimBdOpt over, pc = %d\n", *pc);
  } else if (Row == 1) {
    BdId = ((word) & 0x000000FF);
    // printf("[info]: pc = 0x%08x, %08x, ExecWriteMemBd: {col, row, bd} = {%d,
    // "
    //        "%d, %d}\n",
    //        *pc, word, Col, Row, BdId);
    u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                   (Row << XAIE_ROW_SHIFT) + MEM_DMA_BD0_BASE_ADDR +
                   (BdId * MEM_BD_OFFSET);
#if 0
    opt_write32(BaseAddr, iptr[1]);
    opt_write32(BaseAddr+4u, iptr[2]);
    opt_write32(BaseAddr+8u, iptr[3]);
    opt_write32(BaseAddr+12u, iptr[4]);
    opt_write32(BaseAddr+16u, iptr[5]);
    opt_write32(BaseAddr+20u, iptr[6]);
    opt_write32(BaseAddr+24u, iptr[7]);
    opt_write32(BaseAddr+28u, iptr[8]);
#elif 0
    std::uint32_t *bd = (std::uint32_t *)BaseAddr;
    *(bd++) = iptr[1];
    *(bd++) = iptr[2];
    *(bd++) = iptr[3];
    *(bd++) = iptr[4];
    *(bd++) = iptr[5];
    *(bd++) = iptr[6];
    *(bd++) = iptr[7];
    *(bd++) = iptr[8];
#endif
    *pc += 9;
    // printf("[info] ExecWriteMemBd over, pc = %d\n", *pc);
  } else {
    BdId = ((word) & 0x000000FF);
    // printf("[info]: pc = 0x%08x, %08x, ExecWriteAIEBd: {col, row, bd} = {%d,
    // "
    //        "%d, %d}\n",
    //        *pc, word, Col, Row, BdId);
    u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                   (Row << XAIE_ROW_SHIFT) + AIE_DMA_BD0_BASE_ADDR +
                   (BdId * AIE_BD_OFFSET);
#if 0
      opt_write32(BaseAddr, iptr[1]);
      opt_write32(BaseAddr+4u, iptr[2]);
      opt_write32(BaseAddr+8u, iptr[3]);
      opt_write32(BaseAddr+12u, iptr[4]);
      opt_write32(BaseAddr+16u, iptr[5]);
      opt_write32(BaseAddr+20u, iptr[6]);
#elif 0
    std::uint32_t *bd = (std::uint32_t *)BaseAddr;
    *(bd++) = iptr[1];
    *(bd++) = iptr[2];
    *(bd++) = iptr[3];
    *(bd++) = iptr[4];
    *(bd++) = iptr[5];
    *(bd++) = iptr[6];
#endif
    *pc += 7;
    // printf("[info] ExecWriteAIEBd over, pc = %d\n", *pc);
  }
  // printf("[info] ExecWriteBd over, pc = %d\n", *pc);

  return status;
}

/*
Binarized layout will be
# word0 - READ32 COL X ROW Y
# word1 - REG_ADDR_OFFSET
*/

int32_t ExecRead32(const std::uint32_t *instr_ptr, size_t *const pc,
                   const u64 partBaseAddr, const std::uint32_t word) {

  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);

  const std::uint32_t reg_offset = iptr[1];
  const u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                       (Row << XAIE_ROW_SHIFT) + reg_offset;

  const std::uint32_t *reg_p = (const std::uint32_t *)BaseAddr;

  // printf("[info] ExecRead32 over, pc = %d, word = 0x%x, offset = 0x%x, "
  //        "read_val = 0x%x\n",
  //        *pc, word, reg_offset, *reg_p);
#ifdef __DPU_PERF_PROFILE__
  cru_write32(mmMPIPU_FW_DEBUG_CNT0, word);
  cru_write32(mmMPIPU_FW_DEBUG_CNT0, reg_offset);
  cru_write32(mmMPIPU_FW_DEBUG_CNT0, *reg_p);
#endif

  *pc += OP_READ32_INCR;

  return ERR_SUCCESS;
}

/*
Binarized layout will be
# word0 - READ32POLL COL X ROW Y
# word1 - REG_ADDR_OFFSET
# word2 - REG_CMP_VAL
# word3 - LOOP_CNT
*/

void ExecWriteBdExtendAieTileOpt(const std::uint32_t *instr_ptr, size_t *pc,
                                 u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);

  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t BdId = ((word) & 0x000000FF);
  std::uint8_t ColNum = (((iptr[1]) & 0xFF000000) >> 24);
  std::uint8_t RowNum = (((iptr[1]) & 0x00FF0000) >> 16);

  // printf("[info]: pc = 0x%08x, %08x, ExecWriteBdExtendAieTileOpt: {col, row,
  // "
  //        "bd} = "
  //        "{%d:+%d, %d:+%d, %d}\n",
  //        *pc, *(instr_ptr + *pc), Col, ColNum, Row, RowNum, BdId);

  for (std::uint8_t r = 0u; r < RowNum; r++) {
    for (std::uint8_t c = 0u; c < ColNum; c++) {
      u64 BaseAddr = partBaseAddr + ((Col + c) << XAIE_COL_SHIFT) +
                     ((Row + r) << XAIE_ROW_SHIFT) + AIE_DMA_BD0_BASE_ADDR +
                     (BdId * AIE_BD_OFFSET);
#if 0
      opt_write32(BaseAddr, iptr[2]);
      opt_write32(BaseAddr+4u, iptr[3]);
      opt_write32(BaseAddr+8u, iptr[4]);
      opt_write32(BaseAddr+12u, iptr[5]);
      opt_write32(BaseAddr+16u, iptr[6]);
      opt_write32(BaseAddr+20u, iptr[7]);
#elif 0
      std::uint32_t *bd = (std::uint32_t *)BaseAddr;
      *(bd++) = iptr[2];
      *(bd++) = iptr[3];
      *(bd++) = iptr[4];
      *(bd++) = iptr[5];
      *(bd++) = iptr[6];
      *(bd++) = iptr[7];
#endif
    }
  }
  *pc += 8;
  // printf("[info] ExecWriteBdExtendAieTileOpt over, pc = %d\n", *pc);
}

int32_t ExecWriteBdExtendShimTileOpt(const std::uint32_t *instr_ptr, size_t *pc,
                                     u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);

  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t ColNum = GetInstrTileRow(word);
  std::uint8_t BdId = ((word) & 0x0000000F);
  std::uint8_t DDRType = ((word) & 0x000000F0) >> 4;
  std::uint32_t AddrIncr = iptr[1];
  u64 AddrLow = iptr[3];
  u64 AddrHigh = (iptr[4] & 0x0000FFFF);
  u64 TensorAddr = ((((u64)AddrHigh) << 32) | AddrLow);

  int32_t status = ERR_SUCCESS;

  TensorAddr = MapDDRAddrFromLogicToPhysic(TensorAddr, DDRType, &status);
  // printf("[info]: pc = 0x%08x, %08x, ExecWriteBdExtendShimTileOpt: {col, row,
  // "
  //        "bd, ddr_type, "
  //        "addrIncr, ddr_addr} = {%d:+%d, %d:+%d, %d, %d, %d, %ld}\n",
  //        *pc, word, Col, ColNum, 0, 1, BdId, DDRType, AddrIncr, TensorAddr);
  if (ERR_SUCCESS != status) {
    return status;
  }
  for (std::uint8_t c = 0u; c < ColNum; c++) {
    u64 BaseAddr = partBaseAddr + ((Col + c) << XAIE_COL_SHIFT) +
                   SHIM_DMA_BD0_BASE_ADDR + (BdId * SHIM_BD_OFFSET);
    u64 TensorAddrThis = TensorAddr + c * AddrIncr;

    std::uint32_t tWord0 =
        std::uint32_t((TensorAddrThis) & 0xFFFFFFFFC); // unused 2-LSB
    std::uint32_t tWord1 =
        std::uint32_t((iptr[4] & 0xFFFF0000) | (TensorAddrThis >> 32));

#if 0
    opt_write32(BaseAddr, iptr[2]);
    opt_write32(BaseAddr+4u, tWord0);
    opt_write32(BaseAddr+8u, tWord1);
    opt_write32(BaseAddr+12u, iptr[5]);
    opt_write32(BaseAddr+16u, iptr[6]);
    opt_write32(BaseAddr+20u, iptr[7]);
    opt_write32(BaseAddr+24u, iptr[8]);
    opt_write32(BaseAddr+28u, iptr[9]);
#elif 0
    std::uint32_t *bd = (std::uint32_t *)BaseAddr;
    *(bd++) = iptr[2];
    *(bd++) = tWord0;
    *(bd++) = tWord1;
    *(bd++) = iptr[5];
    *(bd++) = iptr[6];
    *(bd++) = iptr[7];
    *(bd++) = iptr[8];
    *(bd++) = iptr[9];
#endif
  }
  *pc += 10;
  // printf("[info] ExecWriteBdExtendShimTileOpt over, pc = %d\n", *pc);

  return status;
}

void ExecWriteSameBdMemTileOpt(const std::uint32_t *instr_ptr, size_t *pc,
                               u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);

  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t ColNum = GetInstrTileRow(word);
  std::uint8_t Row = 1; // Memtile

  std::uint8_t BdId = ((word) & 0x000000FF);
  // printf("[info]: pc = 0x%08x, %08x, ExecWriteSameBdMemTileOpt: {col, row,
  // bd} "
  //        "= {%d:+%d, %d:+%d, %d}\n",
  //        *pc, word, Col, ColNum, Row, 1, BdId);
  for (std::uint8_t c = 0u; c < ColNum; c++) {
    u64 BaseAddr = partBaseAddr + ((Col + c) << XAIE_COL_SHIFT) +
                   (Row << XAIE_ROW_SHIFT) + MEM_DMA_BD0_BASE_ADDR +
                   (BdId * MEM_BD_OFFSET);

#if 0
    opt_write32(BaseAddr, iptr[1]);
    opt_write32(BaseAddr+4u, iptr[2]);
    opt_write32(BaseAddr+8u, iptr[3]);
    opt_write32(BaseAddr+12u, iptr[4]);
    opt_write32(BaseAddr+16u, iptr[5]);
    opt_write32(BaseAddr+20u, iptr[6]);
    opt_write32(BaseAddr+24u, iptr[7]);
    opt_write32(BaseAddr+28u, iptr[8]);
#elif 0
    std::uint32_t *bd = (std::uint32_t *)BaseAddr;
    *(bd++) = iptr[1];
    *(bd++) = iptr[2];
    *(bd++) = iptr[3];
    *(bd++) = iptr[4];
    *(bd++) = iptr[5];
    *(bd++) = iptr[6];
    *(bd++) = iptr[7];
    *(bd++) = iptr[8];
#endif
  }
  *pc += 9;
  // printf("[info] ExecWriteSameBdMemTileOpt over, pc = %d\n", *pc);
}

void ExecWriteBdExtendMemTileOpt(const std::uint32_t *instr_ptr, size_t *pc,
                                 u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t ColNum = ((word) & 0x000000FF);
  std::uint8_t RowNum = 1;
  std::uint32_t BdListWord = iptr[1];
  std::uint32_t NxtBdListWord = iptr[2];
  std::uint8_t BdList[4];
  std::uint8_t NxtBdList[4];
  for (std::uint8_t c = 0u; c < ColNum; c++) {
    BdList[c] = ((BdListWord >> (c * 8)) & 0x000000FF);
    NxtBdList[c] = ((NxtBdListWord >> (c * 8)) & 0x000000FF);
  }
  // printf("[info]: pc = 0x%08x, %08x, ExecWriteBdExtendMemTile: {col, row, bd,
  // "
  //        "BdList} = {%d:+%d, %d:+%d, %x}\n",
  //        *pc, word, Col, ColNum, Row, RowNum, BdList);

  std::uint32_t BD0_1 = iptr[4];
  for (std::uint8_t c = 0u; c < ColNum; c++) {
    u64 BaseAddr = partBaseAddr + ((Col + c) << XAIE_COL_SHIFT) +
                   (Row << XAIE_ROW_SHIFT) + MEM_DMA_BD0_BASE_ADDR +
                   (BdList[c] * MEM_BD_OFFSET);
    std::uint32_t tWord = ((BD0_1 & 0xFC0FFFFF) | (NxtBdList[c] << 20));
#if 0
    opt_write32(BaseAddr, iptr[3]);
    opt_write32(BaseAddr+4u, tWord);
    opt_write32(BaseAddr+8u, iptr[5]);
    opt_write32(BaseAddr+12u, iptr[6]);
    opt_write32(BaseAddr+16u, iptr[7]);
    opt_write32(BaseAddr+20u, iptr[8]);
    opt_write32(BaseAddr+24u, iptr[9]);
    opt_write32(BaseAddr+28u, iptr[10]);
#elif 0
    std::uint32_t *bd = (std::uint32_t *)BaseAddr;
    *(bd++) = iptr[3];
    *(bd++) = tWord;
    *(bd++) = iptr[5];
    *(bd++) = iptr[6];
    *(bd++) = iptr[7];
    *(bd++) = iptr[8];
    *(bd++) = iptr[9];
    *(bd++) = iptr[10];
#endif
  }
  *pc += 11;
  // printf("[info] ExecWriteBdExtendMemTile over, pc = %d\n", *pc);
}

void ExecWrite32RTPOpt(const std::uint32_t *instr_ptr, size_t *pc,
                       u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  // printf("[info]: pc = %lu\n", *pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);

  u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                 (Row << XAIE_ROW_SHIFT) + iptr[1];
  uint32_t *rtp_base =
      (uint32_t *)(PHYSIC_DDR_ADDR_instruction_start) + iptr[2];

  // printf(
  //     "[info]: pc = 0x%08x, %08x, ExecWrite32RTPOpt: {col, row, offset, val}
  //     = "
  //     "{%d, %d, %08x, %08x}\n",
  //     *pc, word, Col, Row, iptr[2], *rtp_base);
#if 0
  opt_write32(BaseAddr, *rtp_base);
#elif 0
  std::uint32_t *bd = (std::uint32_t *)BaseAddr;
  *(bd) = *rtp_base;
#endif
  *pc += 3;

  // printf("[info] ExecWrite32RTPOpt over, pc = %d\n", *pc);
}

void ExecExtendWrite32Opt(const std::uint32_t *instr_ptr, size_t *pc,
                          u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t ColNum = ((word & 0x000000F0) >> 4);
  std::uint8_t RowNum = ((word & 0x0000000F));
  // printf(
  //     "[info]: pc = 0x%08x, %08x, ExecExtendWrite32Opt: {col, row} = {%d:+%d,
  //     "
  //     "%d:+%d}\n",
  //     *pc, word, Col, ColNum, Row, RowNum);
  for (std::uint8_t r = 0u; r < RowNum; r++) {
    for (std::uint8_t c = 0u; c < ColNum; c++) {
      u64 BaseAddr = partBaseAddr + ((Col + c) << XAIE_COL_SHIFT) +
                     ((Row + r) << XAIE_ROW_SHIFT) + iptr[1];

#if 0
      opt_write32(BaseAddr, iptr[2]);
#elif 0
      std::uint32_t *bd = (std::uint32_t *)BaseAddr;
      *(bd) = iptr[2];
#endif
    }
  }
  *pc += 3;
  // printf("[info] ExecExtendWrite32Opt over, pc = %d\n", *pc);
}

void ExecExtDiffBdWrite32Opt(const std::uint32_t *instr_ptr, size_t *pc,
                             u64 partBaseAddr, std::uint32_t word) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t ColNum = ((word & 0x000000F0) >> 4);
  std::uint8_t RowNum = ((word & 0x0000000F));
  std::uint32_t BdList = iptr[1];
  // printf("[info]: pc = 0x%08x, %08x, ExecExtDiffBdWrite32: {col, row, BdList}
  // "
  //        "= {%d:+%d, %d:+%d, %x}\n",
  //        *pc, word, Col, ColNum, Row, RowNum, BdList);
  std::uint32_t RegVal = iptr[3];
  //   for (std::uint8_t r = 0u; r < RowNum; r++) {
  //     for (std::uint8_t c = 0u; c < ColNum; c++) {
  //       u64 BaseAddr = partBaseAddr + ((Col+c) << XAIE_COL_SHIFT) + ((Row+r)
  //       << XAIE_ROW_SHIFT); std::uint32_t WriteVal = (RegVal & 0xFFFFFFC0) |
  //       ((BdList
  //       >> (c * 8)) & 0x000000FF);
  //     #ifdef __AIESIM__
  //       opt_write32(BaseAddr, WriteVal);
  //     #else
  //       std::uint32_t *bd = (std::uint32_t*)BaseAddr;
  //       *(bd++) = WriteVal;
  //     #endif
  //     }
  //   }
  *pc += 4;
  // printf("[info] ExecExtDiffBdWrite32 over, pc = %d\n", *pc);
}

void ExecRecordTimestamp(const std::uint32_t *instr_ptr, size_t *pc,
                         u64 partBaseAddr, std::uint32_t word) {
#ifndef __AIESIM__
#ifdef _ENABLE_IPU_LX6_
  RecordTimestampImpl((const uint64_t)partBaseAddr, (word & 0x00FFFFFF));
#endif
#endif
  *pc += 1;
}

int32_t ExecWriteShimBdOpt(const std::uint32_t *instr_ptr, size_t *pc,
                           u64 partBaseAddr, std::uint32_t word,
                           bool enable_bo_mc_remap, XAie_DevInst *DevInstp) {
  const std::uint32_t *iptr = instr_ptr + (*pc);

  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t BdId = ((word) & 0x0000000F);
  std::uint8_t DDRType = ((word) & 0x000000F0) >> 4;

  int32_t status = ERR_SUCCESS;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info]: pc = 0x%08x, %08x, ExecWriteShimBdOpt: {col, row, bd, "
  //        "ddr_type} = {%d, %d, %d, %d}\n",
  //        *pc, word, Col, Row, BdId, DDRType);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info]: pc = {}, word = {}, ExecWriteShimBdOpt: "
                           "(col, row, bd, ddr_type) = {} {} {} {}",
                           *pc, word, Col, Row, BdId, DDRType));
#endif

  u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                 (Row << XAIE_ROW_SHIFT) + SHIM_DMA_BD0_BASE_ADDR +
                 (BdId * SHIM_BD_OFFSET);
  u64 AddrLow = iptr[2];
  u64 AddrHigh = (iptr[3] & 0x0000FFFF);
  u64 TensorAddr = ((((u64)AddrHigh) << 32) | AddrLow);
  TensorAddr = MapDDRAddrFromLogicToPhysic(TensorAddr, DDRType, &status);
  if (ERR_SUCCESS != status) {
    return status;
  }

  std::uint32_t tWord0 =
      std::uint32_t((TensorAddr) & 0xFFFFFFFFC); // unused 2-LSB
  std::uint32_t tWord1 =
      std::uint32_t((iptr[3] & 0xFFFF0000) | (TensorAddr >> 32));

  // Create BlockWrite for BDs
  u64 RegOff = (Col << XAIE_COL_SHIFT) + (Row << XAIE_ROW_SHIFT) +
               SHIM_DMA_BD0_BASE_ADDR + (BdId * SHIM_BD_OFFSET);
  XAie_BlockWrite32(DevInstp, RegOff, &iptr[1], 8);

  // Patch the address.
  patch_op_t op;
  op.action = 0;
  op.regaddr = BaseAddr + 4u;
  op.argidx = DDRType;
  if (DDRType == 5 && enable_bo_mc_remap) {
    // This conversion is required for transaction binary to work with
    // transformer xclbin
    op.argidx = 4;
  }

  // std::cout << "TS: Changing ARG idx to " << (std::uint32_t)DDRType
  //           << std::endl;
  op.argplus = TensorAddr;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info]Tejus: DDRType: %d\n", DDRType);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info] Patch BD DDRType: {}", DDRType));
#endif

  XAie_AddCustomTxnOp(DevInstp, XAIE_IO_CUSTOM_OP_DDR_PATCH, (void *)&op,
                      sizeof(op));

#if 0
  opt_write32(BaseAddr, iptr[1]);
  opt_write32(BaseAddr + 4u, tWord0);
  opt_write32(BaseAddr + 8u, tWord1);
  opt_write32(BaseAddr + 12u, iptr[4]);
  opt_write32(BaseAddr + 16u, iptr[5]);
  opt_write32(BaseAddr + 20u, iptr[6]);
  opt_write32(BaseAddr + 24u, iptr[7]);
  opt_write32(BaseAddr + 28u, iptr[8]);
#elif 0
  std::uint32_t *bd = (std::uint32_t *)BaseAddr;
  *(bd++) = iptr[1];
  *(bd++) = tWord0;
  *(bd++) = tWord1;
  *(bd++) = iptr[4];
  *(bd++) = iptr[5];
  *(bd++) = iptr[6];
  *(bd++) = iptr[7];
  *(bd++) = iptr[8];
#endif

  *pc += 9;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info] ExecWriteShimBdOpt over, pc = %d\n", *pc);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info] ExecWriteShimBdOpt over, pc = {}", *pc));
#endif

  return status;
}

void ExecWriteMemBdOpt(const std::uint32_t *instr_ptr, size_t *pc,
                       u64 partBaseAddr, std::uint32_t word,
                       XAie_DevInst *DevInstp) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t BdId = ((word) & 0x000000FF);

#ifdef MC_CONVERSION_DEBUG
  // printf("[info]: pc = 0x%08x, %08x, ExecWriteMemBdOpt: {col, row, bd} = "
  //        "{%d, %d, %d}\n",
  //        *pc, word, Col, Row, BdId);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info]: pc = {}, word = {}, ExecWriteMemBdOpt: "
                           "(col, row, bd) = {} {} {}",
                           *pc, word, Col, Row, BdId));
#endif
  u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                 (Row << XAIE_ROW_SHIFT) + MEM_DMA_BD0_BASE_ADDR +
                 (BdId * MEM_BD_OFFSET);

  u64 RegOff = (Col << XAIE_COL_SHIFT) + (Row << XAIE_ROW_SHIFT) +
               MEM_DMA_BD0_BASE_ADDR + (BdId * MEM_BD_OFFSET);
  XAie_BlockWrite32(DevInstp, RegOff, &iptr[1], 8);

#if 0
  t_write32(BaseAddr, iptr[1]);
  t_write32(BaseAddr + 4u, iptr[2]);
  t_write32(BaseAddr + 8u, iptr[3]);
  t_write32(BaseAddr + 12u, iptr[4]);
  t_write32(BaseAddr + 16u, iptr[5]);
  t_write32(BaseAddr + 20u, iptr[6]);
  t_write32(BaseAddr + 24u, iptr[7]);
  t_write32(BaseAddr + 28u, iptr[8]);
#elif 0
  std::uint32_t *bd = (std::uint32_t *)BaseAddr;
  *(bd++) = iptr[1];
  *(bd++) = iptr[2];
  *(bd++) = iptr[3];
  *(bd++) = iptr[4];
  *(bd++) = iptr[5];
  *(bd++) = iptr[6];
  *(bd++) = iptr[7];
  *(bd++) = iptr[8];
#endif

  *pc += 9;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info] ExecWriteMemBdOpt over, pc = %d\n", *pc);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info] ExecWriteMemBdOpt over, pc = {}", *pc));
#endif
}

void ExecWrite32Opt(const std::uint32_t *instr_ptr, size_t *pc,
                    u64 partBaseAddr, std::uint32_t word,
                    XAie_DevInst *DevInstp) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);

#ifdef MC_CONVERSION_DEBUG
  // printf("[info]: pc = 0x%08x, %08x, ExecWrite32Opt: {col, row} = {%d,
  // %d}\n",
  //        *pc, word, Col, Row);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "[info]: pc = {}, word = {}, ExecWrite32Opt: (col, row) = {}, {}", *pc,
      word, Col, Row));
#endif
  u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                 (Row << XAIE_ROW_SHIFT) + iptr[1];
  u64 RegOff = (Col << XAIE_COL_SHIFT) + (Row << XAIE_ROW_SHIFT) + iptr[1];
  XAie_Write32(DevInstp, RegOff, iptr[2]);

#if 0
  t_write32(BaseAddr, iptr[2]);
#elif 0
  std::uint32_t *bd = (std::uint32_t *)BaseAddr;
  *(bd) = iptr[2];
#endif
  *pc += 3;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info] ExecWrite32Opt over, pc = %d\n", *pc);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("[info] ExecWrite32Opt over, pc = {}", *pc));
#endif
}

int32_t ExecRead32Poll(const std::uint32_t *instr_ptr, size_t *const pc,
                       const u64 partBaseAddr, const std::uint32_t word,
                       XAie_DevInst *DevInstp) {
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word);
  std::uint8_t Row = GetInstrTileRow(word);

  const std::uint32_t reg_offset = iptr[1];
  const u64 BaseAddr = partBaseAddr + (Col << XAIE_COL_SHIFT) +
                       (Row << XAIE_ROW_SHIFT) + reg_offset;
  const std::uint32_t reg_val = iptr[2];
  const std::uint32_t loop_cnt = iptr[3];

  volatile const std::uint32_t *reg_p =
      (volatile const std::uint32_t *)BaseAddr;

  const u64 RegOff =
      (Col << XAIE_COL_SHIFT) + (Row << XAIE_ROW_SHIFT) + reg_offset;
  std::uint32_t Mask = 0xFFFFFFFF; // DPU instruction passes expected reg_val as
                                   // is. So all bits in the mask are set
  std::uint32_t Val = reg_val;

  XAie_MaskPoll(DevInstp, RegOff, Mask, Val, 0xFFFFFFFF);

#ifdef MC_CONVERSION_DEBUG
  // printf("[info] ExecRead32Poll, pc = %d, word = 0x%x, offset = 0x%x, "
  //        "exp_val = 0x%x, loop_cnt = 0x%x\n",
  //        *pc, word, reg_offset, reg_val, loop_cnt);

  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "[info] ExecRead32Poll, pc = {}, word = {}, offset = {}, "
      "exp_val = {}, loop_cnt = {}",
      *pc, word, reg_offset, reg_val, loop_cnt));
#endif

  std::uint32_t loop_idx = 0;
  std::uint32_t read_val = 0;

  //   do{
  //     loop_idx++;
  //     read_val = *reg_p;
  //   } while((loop_idx < loop_cnt) && (read_val != reg_val));

  int32_t status = (read_val == reg_val) ? ERR_SUCCESS : ERR_READ_FAILED;

  // printf("[info] ExecRead32Poll over, exp_val = 0x%x, read_val = 0x%x\n",
  //        reg_val, read_val);

  *pc += OP_READ32_POLL_INCR;

  return status;
}

void ExecSyncOpt(const std::uint32_t *instr_ptr, size_t *pc, std::uint32_t word,
                 std::uint8_t start_col_idx, XAie_DevInst *DevInstp) {
#ifdef __DPU_PERF_PROFILE__
  cru_write32(mmMPIPU_FW_DEBUG_CNT0, start_col_idx << 16 | 0xF002);
#endif
  const std::uint32_t *iptr = instr_ptr + (*pc);
  std::uint8_t Col = GetInstrTileCol(word) + start_col_idx;
  std::uint8_t Row = GetInstrTileRow(word);
  std::uint8_t dir = ((word) & 0x000000FF);
  XAie_DmaDirection Dir = (dir == 0) ? DMA_S2MM : DMA_MM2S;

#ifdef MC_CONVERSION_DEBUG
  // printf("[info]: pc = 0x%08x, %08x, ", *pc, word);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "[info] ExecSyncOpt, pc = {}, word = {}", *pc, word));
#endif
  std::uint32_t config = iptr[1];
  std::uint8_t ChNum = ((config & 0xFF000000) >> 24);
  std::uint8_t ColNum = ((config & 0x00FF0000) >> 16);
  std::uint8_t RowNum = ((config & 0x0000FF00) >> 8);

#ifdef MC_CONVERSION_DEBUG
  // printf("ExecSyncOpt: {col, row, chl, dir} = {%d+%d, %d+%d, %d, %d}\n", Col,
  //        ColNum, Row, RowNum, ChNum, Dir);

  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "ExecSyncOpt: (col, row, chl, dir): {}+{} {}+{} {} {}", Col, ColNum, Row,
      RowNum, ChNum, Dir));
#endif

  tct_op_t op;
  op.word = word;
  op.config = config;

  XAie_AddCustomTxnOp(DevInstp, XAIE_IO_CUSTOM_OP_TCT, (void *)&op, sizeof(op));
  *pc += 2;
}

void ExecMergeSync(const std::uint32_t *instr_ptr, size_t *pc,
                   std::uint32_t word, std::uint32_t start_col_idx,
                   XAie_DevInst *DevInstp) {
#ifdef __DPU_PERF_PROFILE__
  cru_write32(mmMPIPU_FW_DEBUG_CNT0, start_col_idx << 16 | 0xF002);
#endif

  std::uint8_t num_tokens = ((word) & 0x000000FF);
  std::uint8_t num_cols = ((word) & 0x0000FF00) >> 8;

#if STRIX == 1 || STRIX_B0 == 1
  // tile column only [0,3] --->mapping tct 0~3
  uint32_t start_fifo_id = start_col_idx;
#else
  // tile column only [1,4] --->mapping tct 0~3
  uint32_t start_fifo_id = start_col_idx - 1;
#endif

#ifdef MC_CONVERSION_DEBUG
  // printf("ExecSyncMerge num_tokens: %d, num_cols: %d, start_fifo_id: %d\n",
  //        num_tokens, num_cols, start_fifo_id);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "[info]ExecSyncMerge num_tokens: {}, num_cols: {}, start_fifo_id: {}",
      num_tokens, num_cols, start_fifo_id));
//   aie_tct_batch_wait(num_tokens, start_fifo_id, num_cols);
// printf("[info] ExecSyncMerge done waiting\n");
#endif

  tct_op_t op;
  op.word = word;
  op.config = 0;

  XAie_AddCustomTxnOp(DevInstp, XAIE_IO_CUSTOM_OP_MERGE_SYNC, (void *)&op,
                      sizeof(op));
  *pc += OP_MERGESYNC_INCR;

#ifdef __DPU_PERF_PROFILE__
  cru_write32(mmMPIPU_FW_DEBUG_CNT0, start_col_idx << 16 | 0xF003);
#endif
}

XAie_DevInst *initialize_aie_rt(txn_util::device_t device,
                                std::uint32_t num_cols) {

  thread_local XAie_DevInst DevInst = {0};

  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           (std::uint8_t)num_cols,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_CfgInitialize(&DevInst, &ConfigPtr);

  auto RC =
      XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  if (RC) {
    DD_THROW("Failed to start aie-rt in transaction mode");
  }

  return &DevInst;
}

std::vector<std::uint8_t>
txn_util::convert_mc_code(const std::vector<std::uint8_t> &mc_code,
                          device_t device, std::uint32_t num_cols) {

  // dpu instructions
  uint32_t exec_no_op_ = 0;
  uint32_t exec_write_bd_ = 0;
  uint32_t exec_write_shim_bd_ = 0;
  uint32_t exec_mem_bd_ = 0;
  uint32_t exec_write_32_ = 0;
  uint32_t exec_sync_ = 0;
  uint32_t exec_write_tile_bd_ = 0;
  uint32_t exec_write_32_general_ = 0;
  uint32_t exec_write32_extend_shimtile_ = 0;
  uint32_t exec_write32_extend_memtile_ = 0;
  uint32_t exec_write32_extend_diffbd_ = 0;
  uint32_t exec_write32_extend_samebd_memtile_ = 0;
  uint32_t exec_write32_rtp_ = 0;
  uint32_t exec_read_32_ = 0;
  uint32_t exec_read_poll_ = 0;
  uint32_t exec_record_timer_ = 0;
  uint32_t exec_merge_sync_ = 0;

  // Below code is copied from dpufw code.
  size_t pc = 0;
  const size_t limit = mc_code.size() / sizeof(std::uint32_t);
  const std::uint32_t *instr_ptr = (const std::uint32_t *)mc_code.data();

  int32_t status = ERR_SUCCESS;
  auto partBaseAddr = 0;
  auto start_col_idx = 0;

  XAie_DevInst *DevInstp = initialize_aie_rt(device, num_cols);

  const bool enable_bo_mc_remap = true; // in mc_code, this is arg_idx 5
                                        // for TXN API, we will remap this to 4

  while (pc < limit) {
    uint32_t word = *(instr_ptr + pc);
    uint32_t opcode = GetInstrOpcode(word);
#ifdef MC_CONVERSION_DEBUG
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "convert_mc_code: word: {} opcode: {}", word, opcode));
#endif
    switch (opcode) {
    case OP_NOOP:
      exec_no_op_++;
      status = ExecNoOp(&pc);
      break;
    case OP_WRITEBD:
      exec_write_bd_++;
      status = ExecWriteBdOpt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_WRITESHIMBD:
      exec_write_shim_bd_++;
      status = ExecWriteShimBdOpt(instr_ptr, &pc, partBaseAddr, word,
                                  enable_bo_mc_remap, DevInstp);
      break;
    case OP_WRITEMEMBD:
      exec_mem_bd_++;
      ExecWriteMemBdOpt(instr_ptr, &pc, partBaseAddr, word, DevInstp);
      break;
    case OP_WRITE32:
      exec_write_32_++;
      ExecWrite32Opt(instr_ptr, &pc, partBaseAddr, word, DevInstp);
      break;
    case OP_SYNC:
      exec_sync_++;
      ExecSyncOpt(instr_ptr, &pc, word, start_col_idx, DevInstp);
      break;
    case OP_WRITEBD_EXTEND_AIETILE:
      exec_write_tile_bd_++;
      ExecWriteBdExtendAieTileOpt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_WRITE32_EXTEND_GENERAL:
      exec_write_32_general_++;
      ExecExtendWrite32Opt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_WRITEBD_EXTEND_SHIMTILE:
      exec_write32_extend_shimtile_++;
      status = ExecWriteBdExtendShimTileOpt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_WRITEBD_EXTEND_MEMTILE:
      exec_write32_extend_memtile_++;
      ExecWriteBdExtendMemTileOpt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_WRITE32_EXTEND_DIFFBD:
      exec_write32_extend_diffbd_++;
      ExecExtDiffBdWrite32Opt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_WRITEBD_EXTEND_SAMEBD_MEMTILE:
      exec_write32_extend_samebd_memtile_++;
      ExecWriteSameBdMemTileOpt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_DUMPDDR:
#ifdef MC_CONVERSION_DEBUG
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "convert_mc_code::ExecDumpDDR not supported in SimNow!"));
#endif
      pc += 44;
      break;
    case OP_WRITE32_RTP:
      exec_write32_rtp_++;
      ExecWrite32RTPOpt(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_READ32:
      exec_read_32_++;
      status = ExecRead32(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_READ32_POLL:
      exec_read_poll_++;
      status = ExecRead32Poll(instr_ptr, &pc, partBaseAddr, word, DevInstp);
      break;
    case OP_RECORD_TIMESTAMP:
      exec_record_timer_++;
      ExecRecordTimestamp(instr_ptr, &pc, partBaseAddr, word);
      break;
    case OP_MERGESYNC:
      exec_merge_sync_++;
      ExecMergeSync(instr_ptr, &pc, word, start_col_idx, DevInstp);
      break;
    default:
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "convert_mc_code::error: this opcode = {} is not known", opcode));
      DD_THROW(OpsFusion::dd_format(
          "convert_mc_code::error: this opcode = {} is not known", opcode));
      status = ERR_UNKNOWN_INSTR_OP_CODE;
      break;
    } // switch
  }
  auto txn_ptr = XAie_ExportSerializedTransaction(DevInstp, 0, 0U);
  auto hdr = (const XAie_TxnHeader *)txn_ptr;
  std::vector<std::uint8_t> txn_vec(hdr->TxnSize, 0);
  memcpy(txn_vec.data(), txn_ptr, hdr->TxnSize);

  free(txn_ptr);
  XAie_Finish(DevInstp);

  return txn_vec;
}

static constexpr std::uint32_t DMA_BD_NUM = 16;

struct DDRDataStartAddr {
  uint64_t ifmStartAddr;
  uint64_t paramStartAddr;
  uint64_t ofmStartAddr;
  uint64_t interStartAddr;
  DDRDataStartAddr()
      : ifmStartAddr(0), paramStartAddr(0), ofmStartAddr(0), interStartAddr(0) {
  }
};

// #define MC_PATCH_DEBUG

//  patch DDR address，this funtion is from interpreter in LX6.
int32_t patchDDRAddrFromLogicToPhysic(uint32_t &BDData1, uint32_t &BDData2,
                                      const DDRDataStartAddr &DDRAddr) {
  uint32_t addrLow = BDData1;
  uint32_t addrHigh = (BDData2 & 0x00000FFF);
  uint32_t regID = ((BDData2 >> 12) & 0xf);
  uint64_t tensorAddr = ((((uint64_t)addrHigh) << 32) | addrLow);

  switch (regID) {
  case IFM_TYPE:
    tensorAddr += DDRAddr.ifmStartAddr;
#ifdef MC_PATCH_DEBUG
    printf("ifmStartAddr = 0x%llx\n", DDRAddr.ifmStartAddr);
#endif
    break;
  case PARAM_TYPE:
    tensorAddr += DDRAddr.paramStartAddr;
#ifdef MC_PATCH_DEBUG
    printf("paramStartAddr = 0x%llx\n", DDRAddr.paramStartAddr);
#endif
    break;
  case OFM_TYPE:
    tensorAddr += DDRAddr.ofmStartAddr;
#ifdef MC_PATCH_DEBUG
    printf("ofmStartAddr = 0x%llx\n", DDRAddr.ofmStartAddr);
#endif
    break;
  case INTER_TYPE:
    tensorAddr += DDRAddr.interStartAddr;
#ifdef MC_PATCH_DEBUG
    printf("interStartAddr = 0x%llx\n", DDRAddr.interStartAddr);
#endif
    break;
  default:
    break;
  }

  BDData1 =
      static_cast<std::uint32_t>((tensorAddr) & 0xFFFFFFFC); // unused 2-LSB
  BDData2 =
      static_cast<std::uint32_t>((BDData2 & 0xFFFF0000) | (tensorAddr >> 32));
  return 0;
}

uint32_t
patchddrAddress(uint32_t *BDData, uint32_t len, uint32_t addr,
                const DDRDataStartAddr &DDRAddr,
                const std::array<uint32_t, DMA_BD_NUM> &DMABDx2RegAddr) {
  // check if shim tile BD register contains DDR address.
  // This is to support variable number of DMA_BDx register configurations, but
  // this function needs to be checked. Now we write register from DMA_BDx_0 to
  // DMA_BDx_7 every time, for more efficiency, we may only write part of eight
  // DMA_BDx later. One thing to note is that we cannot only write the
  // Base_Address_High of DMA_BDx_2, which also means that the address of
  // DMA_BDx_2 cannot be in the Local Byte Address of control packet(CP). So we
  // start traversing from addr plus 4. Taking DMA_BD0 as an examle, now we
  // fully configure from 0x1D000 to 0x1D01C, later we may only config five
  // registers, say from 0x1D00C to 0x1D01C. the position of Base_Address_High
  // in BD data is variable, and may even not exist. so We need to check if the
  // shim tile DMA_BDx register contains the DDR address.
  for (std::uint32_t i = 1; i < len + 1; i++) {
    addr += 4;
    if (DMABDx2RegAddr.end() !=
        std::find(DMABDx2RegAddr.begin(), DMABDx2RegAddr.end(), addr)) {
      // patch DDR Addrese from offset to phisical address
      patchDDRAddrFromLogicToPhysic(BDData[i - 1], BDData[i], DDRAddr);
    }
  }

  return 0;
}

void txn_util::pactch_mc_control_packet(
    uint32_t *mc_control_packet, size_t mc_control_packet_size_bytes,
    uint64_t ddr_base_ifm, uint64_t ddr_base_param, uint64_t ddr_base_ofm,
    uint64_t ddr_base_inter, bool pad_control_packet) {
  if (0 == mc_control_packet_size_bytes) {
    return;
  }

  if (nullptr == mc_control_packet) {
    return;
  }

  DDRDataStartAddr DDRAddr;
  DDRAddr.ifmStartAddr = ddr_base_ifm;
  DDRAddr.paramStartAddr = ddr_base_param;
  DDRAddr.ofmStartAddr = ddr_base_ofm;
  DDRAddr.interStartAddr = ddr_base_inter;

  // list all shim tile BD registers DDR address need to be processed
  std::array<uint32_t, DMA_BD_NUM> DMABDx2RegAddr;
  for (std::uint32_t i = 0; i < 16; i++) {
    DMABDx2RegAddr[i] = 0x0001D008 + 0x20 * i;
  }

  uint32_t dataSize = 0;
  uint32_t localByteAddress = 0;
  size_t pc = 0;
  // Traverse all mc code ddr instructions
  //// mc_control_packet_size_bytes is in bytes, pc is in words. So divide by 4
  while (pc < mc_control_packet_size_bytes / 4) {
    // read packet header and control packet, parse the data size and BD
    // register addr
    pc += 2;
    dataSize = ((mc_control_packet[pc - 1] >> 20) & 0x3);
    localByteAddress = (mc_control_packet[pc - 1] & 0xfffff);

    // patch shim tile register DMA_BDx DDR address
    patchddrAddress(&mc_control_packet[pc], dataSize, localByteAddress, DDRAddr,
                    DMABDx2RegAddr);
    pc += (dataSize + 1);

    // control packets aligned to 256 bits
    if (pad_control_packet) {
      pc += (8 - (pc % 8)) % 8;
    }
  }
}

} // namespace utils

transaction_op::transaction_op(const std::vector<uint8_t> &txn) {
  txn_op_.resize(txn.size() + TXN_OP_SIZE);

  XAie_TxnHeader *hdr = (XAie_TxnHeader *)txn.data();

  uint32_t *ptr = (uint32_t *)txn_op_.data();
  // set op code
  *ptr = TXN_OP_CODE;
  ptr++;
  *ptr = hdr->TxnSize + TXN_OP_SIZE;

  memcpy(txn_op_.data() + TXN_OP_SIZE, txn.data(), txn.size());
}

size_t transaction_op::get_txn_instr_size() {
  uint32_t *ptr = (uint32_t *)txn_op_.data();
  return *(++ptr);
}

std::vector<uint8_t> transaction_op::get_txn_op() { return txn_op_; }

size_t transaction_op::getInstrBufSize(const std::string &txn_str) {
  return TXN_OP_SIZE + txn_str.size();
}
void transaction_op::addTxnOp(const std::string &txn_str, void *instr_buf) {

  XAie_TxnHeader *hdr = (XAie_TxnHeader *)txn_str.data();

  uint32_t *ptr = (uint32_t *)instr_buf;
  // set op code
  *ptr = TXN_OP_CODE;
  ptr++;
  *ptr = hdr->TxnSize + TXN_OP_SIZE;

  uint8_t *instr_ptr = (uint8_t *)instr_buf;

  memcpy(instr_ptr + TXN_OP_SIZE, txn_str.data(), txn_str.size());
}
