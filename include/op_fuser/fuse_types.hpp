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

#pragma once

#include <any>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <utils/utils.hpp>
#include <vector>

struct CtrlPktPatchInfo {
  size_t
      offset;  // offset in ctrl pkt bin at which BD address has to be modified
  size_t size; // number of bytes to modify
  size_t xrt_arg_idx;
  size_t bo_offset; // BD address(at the offset) to be updated with this value
                    // and physical address.
};

namespace OpsFusion {

struct OpPDIMap {
  using op_type_t = std::string;
  using pdi_id_t = std::uint32_t;
  using kernel_id_t = std::string;

  using OpPDIInfoMap = std::map<op_type_t, pdi_id_t>;
  using PDIKernelInfoMap = std::map<pdi_id_t, kernel_id_t>;

  OpPDIInfoMap op_to_pdi_id_map;
  PDIKernelInfoMap pdi_id_to_kernel_map;
};

struct Partition {
  // describes [start, end) interval
  std::pair<size_t, size_t> op_range;
  uint8_t pdi_id;
};

struct Metadata {
  uint32_t major_version;
  uint32_t minor_version;
  struct OpInfo {
    std::string name;
    std::string type;
    std::vector<std::string> in_args;
    std::vector<std::string> const_args;
    std::vector<std::string> out_args;
    std::vector<uint8_t> txn_bin;
    std::vector<CtrlPktPatchInfo> ctrl_pkt_patch_info;
    std::map<std::string, std::any> attr;
    std::uint8_t pdi_id = 0;
  };
  struct TensorInfo {
    size_t size;
    size_t arg_idx; // TODO : Should be renamed xrt_arg_idx
    std::vector<std::string> packed_tensors;
  };
  struct OffsetInfo {
    std::string parent_name; // Parent packed_tensor's name
    size_t offset;           // Offset in the parent tensor
    size_t arg_idx;          // TODO : Should be renamed xrt_arg_idx
    std::string dtype;
    std::vector<size_t> shape;
    size_t size_in_bytes; // Final size as per the kernel's reqs.
    std::string file_name;
    size_t file_size;
  };

  struct Span {
    size_t offset;
    size_t size;
  };

  std::vector<OpInfo> op_list;
  std::map<std::string, TensorInfo>
      fused_tensors; // fused_tensor.name --> TensorInfo
  std::map<std::string, OffsetInfo>
      tensor_map;                              // onnxtensor.name --> OffsetInfo
  std::map<std::string, Span> super_instr_map; // op.name --> Op's super buffer
  std::map<std::string, Span> const_map;       // op.name --> Op's const buffer
  std::map<std::string, Span> ctrl_pkt_map; // op.name --> Op's ctrl pkt buffer
  std::set<std::string>
      scratch_op_set; // set of ops which require internal scratch pad
  size_t max_op_scratch_pad_size; // max internal scratch pad for all op
  size_t max_tensor_padding_sz;   // max padding for input tensor of op

  // Placeholder to keep any extra info
  std::map<std::string, std::any> aux_info;

  std::string json_path;

  // Information on PDI partitioning
  std::vector<Partition> partitions;
};

static const std::set<std::string> CONTROL_OPS{"PM_LOAD", "RECORD_TIMER",
                                               "PREEMPTION"};

static constexpr uint32_t CONTROL_PDI_ID = 0xFF;

uint8_t static inline get_pdi_id(const OpPDIMap &op_pdi_map,
                                 const std::string &op_type) {
  if (op_pdi_map.op_to_pdi_id_map.empty()) {
    return 0;
  }

  uint8_t pdi_id = 0;

  if (OpsFusion::CONTROL_OPS.find(op_type) != OpsFusion::CONTROL_OPS.end()) {
    // these ops can be supported on any PDI
    return OpsFusion::CONTROL_PDI_ID;
  }

  if (op_pdi_map.op_to_pdi_id_map.find(op_type) !=
      op_pdi_map.op_to_pdi_id_map.end()) {
    pdi_id = op_pdi_map.op_to_pdi_id_map.at(op_type);
  } else {
    throw std::runtime_error("Op type " + op_type +
                             " not registered in op_pdi_map");
  }

  return pdi_id;
}

static std::vector<std::string> get_op_args(const Metadata::OpInfo &op_info) {
  return Utils::concat_vectors(op_info.in_args, op_info.const_args,
                               op_info.out_args);
}

static size_t get_op_args_count(const Metadata::OpInfo &op_info) {
  return (op_info.in_args.size() + op_info.const_args.size() +
          op_info.out_args.size());
}

} // namespace OpsFusion
