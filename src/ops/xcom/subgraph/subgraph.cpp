/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <any>
#include <iostream>
#include <map>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <experimental/xrt_error.h>

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/op_interface.hpp>
// #include <ops/ops_common.hpp>
#include <ops/xcom/subgraph/subgraph.hpp>
#include <utils/logging.hpp>

#include "txn/txn_utils.hpp"

namespace ryzenai {
namespace xcom {

const std::string XCOM_SG_4x4_XCLBIN_PATH = "/xclbin/stx/4x4_dpu_sg.xclbin";
subgraph::subgraph(bool load_xrt) {

  auto XCLBIN_FNAME = OpInterface::get_dd_base_dir() + XCOM_SG_4x4_XCLBIN_PATH;
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  }
}

std::vector<OpArgMap>
subgraph::get_buffer_reqs(std::vector<Tensor> &input,
                          std::vector<Tensor> &output,
                          const std::map<std::string, std::any> &attr) const {

  DD_THROW_IF(attr.find("ifm_size") == attr.end(),
              "Can't find ifm_size attribute for the subgraph");
  size_t act_size = std::any_cast<size_t>(attr.find("ifm_size")->second);

  DD_THROW_IF(attr.find("param_size") == attr.end(),
              "Can't find param_size attribute for the subgraph");
  size_t params_size = std::any_cast<size_t>(attr.find("param_size")->second);

  DD_THROW_IF(attr.find("padded_ofm_tile_shape") == attr.end(),
              "Can't find padded_ofm_tile_shape attribute for the subgraph");
  DD_THROW_IF(attr.find("out_dtypes") == attr.end(),
              "Can't find out_dtypes attribute for the subgraph");

  std::vector<size_t> padded_ofm_tile_shape =
      std::any_cast<std::vector<size_t>>(
          attr.find("padded_ofm_tile_shape")->second);
  std::vector<std::string> out_dtypes =
      std::any_cast<std::vector<std::string>>(attr.find("out_dtypes")->second);

  size_t out_size = std::accumulate(padded_ofm_tile_shape.begin(),
                                    padded_ofm_tile_shape.end(), size_t{1},
                                    std::multiplies{}) *
                    Utils::get_size_of_type(out_dtypes.at(0));

  DD_THROW_IF(attr.find("inter_size") == attr.end(),
              "Can't find inter_size attribute for the subgraph");
  size_t inter_size = std::any_cast<size_t>(attr.find("inter_size")->second);

  DD_THROW_IF(attr.find("num_tiles") == attr.end(),
              "Can't find num_tiles attribute for the subgraph")
  size_t num_tiles = std::any_cast<size_t>(attr.find("num_tiles")->second);

  DD_THROW_IF(0 == num_tiles, "Number of tiles cannot be 0 for the subgraph");

  out_size *= num_tiles;

  DD_THROW_IF(attr.find("ifm_addr") == attr.end(),
              "Can't find ifm_addr attribute for the subgraph");
  const size_t ifm_addr = std::any_cast<size_t>(attr.find("ifm_addr")->second);

  // std::cout << "Input size: " << act_size << std::endl;
  // std::cout << "Params size: " << params_size << std::endl;
  // std::cout << "Output size: " << out_size << std::endl;
  // std::cout << "Intermediate size: " << inter_size << std::endl;
  // std::cout << "num_tiles: " << num_tiles << std::endl;
  // std::cout << "ifm_addr: " << ifm_addr << std::endl;

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, act_size, ifm_addr},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 0, 0, params_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 0, 0, out_size},
      {OpArgMap::OpArgType::SCRATCH_PAD, 3, 0, 0, inter_size},
  };

  return arg_map;
}

void subgraph::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // const inputs
  constexpr std::uint32_t params_idx = 0;
  auto params_shape = const_params.at(params_idx).shape;
  auto params_size = std::accumulate(params_shape.begin(), params_shape.end(),
                                     size_t{1}, std::multiplies{}) *
                     Utils::get_size_of_type(const_params.at(params_idx).dtype);

  DD_THROW_IF(attr.find("param_addr") == attr.end(),
              "Can't find param_addr attribute for the subgraph");
  size_t offset = std::any_cast<size_t>(attr.find("param_addr")->second);

  io.write(offset, const_params.at(params_idx).data, params_size);
}

void subgraph::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  auto PARAM_BO_SHAPE = const_params.at(0).shape;
  auto PARAM_BO_SIZE =
      std::accumulate(PARAM_BO_SHAPE.begin(), PARAM_BO_SHAPE.end(), size_t{1},
                      std::multiplies{});

  param_bo =
      xrt::bo(xrt_ctx_->get_device(), PARAM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(2));

  param_bo.write(const_params.at(0).data);
  param_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::string path = "";
  setup_instruction_bo(path);
}

void subgraph::execute(const std::vector<Tensor> &input,
                       std::vector<Tensor> &output) {
  auto IN_BO_SHAPE = input.at(0).shape;
  auto OUT_BO_SHAPE = output.at(0).shape;
  auto IN_BO_SIZE = std::accumulate(IN_BO_SHAPE.begin(), IN_BO_SHAPE.end(),
                                    size_t{1}, std::multiplies{}) +
                    2112;
  auto OUT_BO_SIZE = std::accumulate(OUT_BO_SHAPE.begin(), OUT_BO_SHAPE.end(),
                                     size_t{1}, std::multiplies{});

  ifm_bo = xrt::bo(xrt_ctx_->get_device(), IN_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofm_bo = xrt::bo(xrt_ctx_->get_device(), OUT_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));

  // auto dummy_bo_1 = xrt::bo(xrt_ctx_->get_device(), 4096,
  // XRT_BO_FLAGS_HOST_ONLY,
  //                     xrt_ctx_->get_kernel().group_id(8));
  // auto dummy_bo_2 = xrt::bo(xrt_ctx_->get_device(), 4096,
  // XRT_BO_FLAGS_HOST_ONLY,
  //                     xrt_ctx_->get_kernel().group_id(8));

  ifm_bo.write(input.at(0).data);
  ofm_bo.write(output.at(0).data);
  ifm_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ofm_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::run run;
  auto kernel = xrt_ctx_->get_kernel();
  run = kernel(2, instr_bo, instr_bo.size() / sizeof(int),
               ifm_bo.address() + DDR_AIE_ADDR_OFFSET,
               param_bo.address() + DDR_AIE_ADDR_OFFSET,
               ofm_bo.address() + DDR_AIE_ADDR_OFFSET);

  try {
    run.wait2();
  } catch (const std::exception &e) {
    std::cerr << "Details: " << e.what() << std::endl;

    xrt::error err(xrt_ctx_->get_device(), XRT_ERROR_CLASS_AIE);
    if (err.get_error_code()) {
      std::string err_message = std::string("Error while executing pdi_id: ") +
                                ", info: " + err.to_string();
      std::cerr << err_message << std::endl;
      //   RYZENAI_LOG_TRACE(err_message);
    }
  }

  ofm_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofm_bo.read(output.at(0).data);
}

const std::vector<uint8_t> subgraph::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  DD_THROW_IF(attr.find("mc_code_bin") == attr.end(),
              "Can't find mc_code_bin attribute for the subgraph");
  DD_THROW_IF(attr.find("num_cols_design") == attr.end(),
              "Can't find num_cols_design attribute for the subgraph");

  std::vector<uint8_t> mc_code_bin =
      std::any_cast<std::vector<uint8_t>>(attr.find("mc_code_bin")->second);
  size_t num_cols_design =
      std::any_cast<size_t>(attr.find("num_cols_design")->second);
  std::vector<uint8_t> txn_bin = utils::txn_util::convert_mc_code(
      mc_code_bin, utils::txn_util::device_t::RYZENAI_STX,
      static_cast<uint32_t>(num_cols_design));
  // std::cout << "txn bin size: " << txn_bin.size() << ", num_cols_design: " <<
  // num_cols_design << std::endl;

  return txn_bin;
}

const std::vector<uint8_t> subgraph::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

void subgraph::setup_instruction_bo(std::string &txn_fname) {
  // auto txn =
  // OpInterface::get_dd_base_dir() +
  // "/tests/xcom_subgraphs/0621_conv_48x32_cases/case_full/mc_code_txn.bin";
  auto txn =
      OpInterface::get_dd_base_dir() +
      "/tests/xcom_subgraphs/0610_conv_32x32_cases/case_full/mc_code_txn.bin";
  auto txn_bin = OpsFusion::read_bin_file(txn);
  auto txn_bin_u8 = std::vector<uint8_t>(txn_bin.begin(), txn_bin.end());
  auto i_buf = transaction_op(static_cast<std::vector<uint8_t>>(txn_bin_u8));
  size_t instr_bo_words = i_buf.get_txn_instr_size();
  instr_bo =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_words,
              xrt::bo::flags::cacheable, xrt_ctx_->get_kernel().group_id(1));
  instr_bo.write(i_buf.get_txn_op().data());
  instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "Instuction BO setup" << std::endl;
}

void subgraph::format_output(const Tensor &out_tensor, void *hw_out_ptr,
                             size_t sz, size_t tensor_idx,
                             const std::map<std::string, std::any> &attr) {

  DD_THROW_IF(
      attr.find("padded_ofm_tile_offset_dims") == attr.end(),
      "Can't find padded_ofm_tile_offset_dims attribute for the subgraph");
  DD_THROW_IF(attr.find("ofm_tile_concat_shapes") == attr.end(),
              "Can't find ofm_tile_concat_shapes attribute for the subgraph");
  DD_THROW_IF(attr.find("padded_ofm_tile_shape") == attr.end(),
              "Can't find padded_ofm_tile_shape attribute for the subgraph");
  DD_THROW_IF(attr.find("ofm_tile_shape") == attr.end(),
              "Can't find ofm_tile_shape attribute for the subgraph");

  const std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims =
      std::any_cast<std::vector<std::vector<size_t>>>(
          attr.find("padded_ofm_tile_offset_dims")->second);

  const std::vector<std::vector<size_t>> ofm_tile_concat_shapes =
      std::any_cast<std::vector<std::vector<size_t>>>(
          attr.find("ofm_tile_concat_shapes")->second);

  DD_THROW_IF(padded_ofm_tile_offset_dims.size() !=
                  ofm_tile_concat_shapes.size(),
              "OFM offset and size num elements mismatch");

  const std::vector<size_t> padded_ofm_tile_shape =
      std::any_cast<std::vector<size_t>>(
          attr.find("padded_ofm_tile_shape")->second);
  const std::vector<size_t> ofm_tile_shape =
      std::any_cast<std::vector<size_t>>(attr.find("ofm_tile_shape")->second);

  DD_THROW_IF(padded_ofm_tile_shape.size() != ofm_tile_shape.size(),
              "mismatch in size");
  DD_THROW_IF(padded_ofm_tile_shape.size() != 3,
              "Expect 3 dim vector for padded shape");

  size_t padded_ofm_tile_size = std::accumulate(padded_ofm_tile_shape.begin(),
                                                padded_ofm_tile_shape.end(),
                                                size_t{1}, std::multiplies{}) *
                                Utils::get_size_of_type(out_tensor.dtype);

  DD_THROW_IF(Utils::get_size_of_type(out_tensor.dtype) != 1,
              "format output assumes 1 byte per datum");
  // assumes data format is int8/uint8
  // todo: bring in avx depad code here
  uint8_t *dest = (uint8_t *)out_tensor.data;

  // constexpr size_t c_stride = 1;
  const size_t w_stride = padded_ofm_tile_shape.at(2);
  const size_t h_stride =
      padded_ofm_tile_shape.at(1) * padded_ofm_tile_shape.at(2);

  for (size_t i = 0; i < padded_ofm_tile_offset_dims.size(); i++) {
    size_t src_offset = i * padded_ofm_tile_size;
    uint8_t *src = (uint8_t *)hw_out_ptr + src_offset;

    for (size_t h_idx = padded_ofm_tile_offset_dims.at(i).at(0);
         h_idx < padded_ofm_tile_offset_dims.at(i).at(0) +
                     ofm_tile_concat_shapes.at(i).at(0);
         h_idx++) {
      for (size_t w_idx = padded_ofm_tile_offset_dims.at(i).at(1);
           w_idx < padded_ofm_tile_offset_dims.at(i).at(1) +
                       ofm_tile_concat_shapes.at(i).at(1);
           w_idx++) {
        std::memcpy(dest,
                    &src[h_idx * h_stride + w_idx * w_stride +
                         padded_ofm_tile_offset_dims.at(i).at(2)],
                    ofm_tile_concat_shapes.at(i).at(2));
        dest += ofm_tile_concat_shapes.at(i).at(2);
        // for(size_t c_idx = padded_ofm_tile_offset_dims.at(i).at(2); c_idx <
        // padded_ofm_tile_offset_dims.at(i).at(2) +
        // ofm_tile_concat_shapes.at(i).at(2); c_idx++){
        //   *dest++ = src[h_idx*h_stride + w_idx*w_stride + c_idx];
        // }
      }
    }
  }
}
} // namespace xcom
} // namespace ryzenai
