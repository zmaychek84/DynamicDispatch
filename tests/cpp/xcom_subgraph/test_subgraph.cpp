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

#include <iostream>

#include <op_fuser/fusion_rt.hpp>
#include <ops/op_interface.hpp>
#include <ops/xcom/subgraph/subgraph.hpp>

int conv32x32_single_tile(uint32_t num_exec_loops) {
  std::cout << "Running 32x32 single tile subgraph test" << std::endl;

  auto base_dir = OpInterface::get_dd_base_dir() +
                  "/tests/xcom_subgraphs/0610_conv_32x32_cases/case_full/";
  auto param = OpsFusion::read_bin_file(base_dir + "param.bin");
  auto ifm = OpsFusion::read_bin_file(base_dir + "ifm.bin");
  auto golden =
      OpsFusion::read_bin_file(base_dir + "golden_Conv2d__Conv2d_ret_fix.bin");

  auto txn_bin = OpsFusion::read_bin_file(base_dir + "mc_code_txn.bin");
  std::vector<std::uint8_t> mc_code_bin =
      OpsFusion::read_bin_file<std::uint8_t>(base_dir + "mc_code.bin");
  std::vector<int8_t> ofm(golden.size(), 0);
  std::vector<Tensor> input;
  std::vector<Tensor> params;
  std::vector<Tensor> output;

  struct Tensor ifm_ten = {ifm.data(), {1, 32, 32, 64}, "int8"};
  input.push_back(ifm_ten);
  struct Tensor param_ten = {param.data(), {1, param.size()}, "int8"};
  params.push_back(param_ten);
  struct Tensor ofm_ten = {ofm.data(), {1, 32, 32, 64}, "int8"};
  output.push_back(ofm_ten);

  std::vector<Tensor> meta_tensor;
  meta_tensor.push_back(ifm_ten);
  meta_tensor.push_back(param_ten);
  meta_tensor.push_back(ofm_ten);

  // ryzenai::xcom::subgraph sg(true);
  // sg.initialize_const_params(params, {});
  // sg.execute(input, output);

  auto xclbin =
      OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu_sg.xclbin";
  auto DPU_KERNEL_NAME = "DPU";
  constexpr std::uint32_t num_contexts = 1;

  auto dd_rt = std::make_unique<OpsFusion::FusionSubgraphRuntime>(
      xclbin, DPU_KERNEL_NAME, num_contexts);

  std::vector<uint8_t> txn(txn_bin.size());
  std::memcpy(txn.data(), txn_bin.data(), txn_bin.size());
  std::map<std::string, std::any> attr;

  std::vector<size_t> padded_ifm_tile_offsets = {0};

  std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims = {{0, 0, 0}};

  std::vector<std::vector<size_t>> ofm_tile_concat_shapes = {{32, 32, 64}};

  std::vector<size_t> ifm_tile_shape = {32, 32, 64};
  std::vector<std::string> in_dtypes = {"int8"};
  std::vector<std::string> out_dtypes = {"int8"};

  attr["mc_code_bin"] = mc_code_bin;
  attr["txn_bin"] = txn;
  attr["num_tiles"] = static_cast<size_t>(1);
  attr["padded_ifm_tile_offsets"] = padded_ifm_tile_offsets;
  attr["padded_ofm_tile_offset_dims"] = padded_ofm_tile_offset_dims;
  attr["ofm_tile_concat_shapes"] = ofm_tile_concat_shapes;
  attr["ifm_tile_shape"] = ifm_tile_shape;
  attr["padded_ifm_tile_shape"] = ifm_tile_shape;
  attr["ofm_tile_shape"] = ifm_tile_shape;
  attr["padded_ofm_tile_shape"] = ifm_tile_shape;
  attr["ifm_addr"] = static_cast<size_t>(0);
  attr["ifm_size"] = static_cast<size_t>(65536);
  attr["param_addr"] = static_cast<size_t>(0);
  attr["param_size"] = static_cast<size_t>(4480);
  attr["ofm_addr"] = static_cast<size_t>(0);
  attr["ofm_size"] = static_cast<size_t>(65536);
  attr["inter_addr"] = static_cast<size_t>(0);
  attr["inter_size"] = static_cast<size_t>(2097152);
  attr["ofm_orig_tile_size"] = static_cast<size_t>(32 * 32 * 64);
  attr["num_cols_design"] = static_cast<size_t>(4);
  attr["in_dtypes"] = in_dtypes;
  attr["out_dtypes"] = out_dtypes;
  attr["subgraph_name"] = static_cast<std::string>("conv32x32_single_tile");

  // std::vector<uint8_t> &txn =
  // reinterpret_cast<std::vector<uint8_t>&>(txn_bin);
  dd_rt->init(params, attr);

  for (uint32_t i = 0; i < num_exec_loops; i++) {
    dd_rt->execute_subgraph(input, output);
  }

  int err_count = 0;
  for (int i = 0; i < golden.size(); i++) {
    if (golden.at(i) != ofm.at(i)) {
      std::cout << "golden, actual : " << std::to_string(golden.at(i)) << ", "
                << std::to_string(ofm.at(i)) << std::endl;
      err_count++;
      getchar();
    }
  }

  if (err_count != 0) {
    std::cout << "conv32x32_single_tile::Test failed with error count : "
              << err_count << std::endl;
    return -1;
  }

  std::cout << "conv32x32_single_tile::TEST PASS" << std::endl;

  return 0;
}

int conv48x32_single_tile(uint32_t num_exec_loops) {
  std::cout << "Running 48x32 single tile subgraph test" << std::endl;

  auto base_dir = OpInterface::get_dd_base_dir() +
                  "/tests/xcom_subgraphs/0621_conv_48x32_cases/case_full/";
  auto param = OpsFusion::read_bin_file(base_dir + "param.bin");
  auto ifm = OpsFusion::read_bin_file(base_dir + "ifm.bin");
  auto golden = OpsFusion::read_bin_file(
      base_dir + "golden_Sequential__Sequential_Conv2d_0__ret_fix.bin");
  auto txn_bin = OpsFusion::read_bin_file(base_dir + "mc_code_txn.bin");
  std::vector<std::uint8_t> mc_code_bin =
      OpsFusion::read_bin_file<std::uint8_t>(base_dir + "mc_code.bin");

  std::vector<int8_t> ofm(golden.size(), 0);
  std::vector<Tensor> input;
  std::vector<Tensor> params;
  std::vector<Tensor> output;

  struct Tensor ifm_ten = {ifm.data(), {1, 48, 32, 64}, "int8"};
  input.push_back(ifm_ten);
  struct Tensor param_ten = {param.data(), {1, param.size()}, "int8"};
  params.push_back(param_ten);
  struct Tensor ofm_ten = {ofm.data(), {1, 48, 32, 64}, "int8"};
  output.push_back(ofm_ten);

  std::vector<Tensor> meta_tensor;
  meta_tensor.push_back(ifm_ten);
  meta_tensor.push_back(param_ten);
  meta_tensor.push_back(ofm_ten);

  // ryzenai::xcom::subgraph sg(true);
  // sg.initialize_const_params(params, {});
  // sg.execute(input, output);

  std::vector<uint8_t> txn(txn_bin.size());
  std::memcpy(txn.data(), txn_bin.data(), txn_bin.size());
  std::map<std::string, std::any> attr;
  std::vector<size_t> padded_ifm_tile_offsets = {0};

  std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims = {{0, 0, 0}};

  std::vector<std::vector<size_t>> ofm_tile_concat_shapes = {{48, 32, 64}};

  std::vector<size_t> ifm_tile_shape = {48, 32, 64};
  std::vector<std::string> in_dtypes = {"int8"};
  std::vector<std::string> out_dtypes = {"int8"};

  attr["mc_code_bin"] = mc_code_bin;
  attr["txn_bin"] = txn;
  attr["num_tiles"] = static_cast<size_t>(1);
  attr["padded_ifm_tile_offsets"] = padded_ifm_tile_offsets;
  attr["padded_ofm_tile_offset_dims"] = padded_ofm_tile_offset_dims;
  attr["ofm_tile_concat_shapes"] = ofm_tile_concat_shapes;
  attr["ifm_tile_shape"] = ifm_tile_shape;
  attr["padded_ifm_tile_shape"] = ifm_tile_shape;
  attr["ofm_tile_shape"] = ifm_tile_shape;
  attr["padded_ofm_tile_shape"] = ifm_tile_shape;
  attr["ifm_addr"] = static_cast<size_t>(2112);
  attr["ifm_size"] = static_cast<size_t>(102528);
  attr["param_addr"] = static_cast<size_t>(0);
  attr["param_size"] = static_cast<size_t>(37248);
  attr["ofm_addr"] = static_cast<size_t>(0);
  attr["ofm_size"] = static_cast<size_t>(98336);
  attr["inter_addr"] = static_cast<size_t>(0);
  attr["inter_size"] = static_cast<size_t>(2097152);
  attr["ofm_orig_tile_size"] = static_cast<size_t>(48 * 32 * 64);
  attr["num_cols_design"] = static_cast<size_t>(4);
  attr["in_dtypes"] = in_dtypes;
  attr["out_dtypes"] = out_dtypes;
  attr["subgraph_name"] = static_cast<std::string>("conv48x32_single_tile");

  auto xclbin =
      OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu_sg.xclbin";
  auto DPU_KERNEL_NAME = "DPU";
  constexpr std::uint32_t num_contexts = 1;

  auto dd_rt = std::make_unique<OpsFusion::FusionSubgraphRuntime>(
      xclbin, DPU_KERNEL_NAME, num_contexts);
  // std::vector<uint8_t> &txn =
  // reinterpret_cast<std::vector<uint8_t>&>(txn_bin);
  dd_rt->init(params, attr);
  for (uint32_t i = 0; i < num_exec_loops; i++) {
    dd_rt->execute_subgraph(input, output);
  }

  int err_count = 0;
  for (int i = 0; i < golden.size(); i++) {
    if (golden.at(i) != ofm.at(i)) {
      std::cout << "golden, actual : " << std::to_string(golden.at(i)) << ", "
                << std::to_string(ofm.at(i)) << std::endl;
      err_count++;
      getchar();
    }
  }

  if (err_count != 0) {
    std::cout << "conv48x32_single_tile::Test failed with error count : "
              << err_count << std::endl;
    return -1;
  }

  std::cout << "conv48x32_single_tile::TEST PASS" << std::endl;

  return 0;
}

int conv32x32_two_tiles(uint32_t num_exec_loops, uint32_t num_contexts) {
  std::cout << "Running 32x32 two tile subgraph test" << std::endl;

  auto base_dir = OpInterface::get_dd_base_dir() +
                  "/tests/xcom_subgraphs/0610_conv_32x32_cases/case_half/";
  auto golden_base_dir =
      OpInterface::get_dd_base_dir() +
      "/tests/xcom_subgraphs/0610_conv_32x32_cases/case_full/";
  auto param = OpsFusion::read_bin_file(base_dir + "param.bin");
  auto ifm = OpsFusion::read_bin_file(golden_base_dir + "ifm.bin");
  auto golden = OpsFusion::read_bin_file(golden_base_dir +
                                         "golden_Conv2d__Conv2d_ret_fix.bin");

  auto txn_bin = OpsFusion::read_bin_file(base_dir + "mc_code_txn.bin");
  std::vector<std::uint8_t> mc_code_bin =
      OpsFusion::read_bin_file<std::uint8_t>(base_dir + "mc_code.bin");

  std::vector<int8_t> ofm(golden.size(), 0);
  std::vector<Tensor> input;
  std::vector<Tensor> params;
  std::vector<Tensor> output;

  struct Tensor ifm_ten = {ifm.data(), {1, 32, 32, 64}, "int8"};
  input.push_back(ifm_ten);
  struct Tensor param_ten = {param.data(), {1, param.size()}, "int8"};
  params.push_back(param_ten);
  struct Tensor ofm_ten = {ofm.data(), {1, 32, 32, 64}, "int8"};
  output.push_back(ofm_ten);

  std::vector<Tensor> meta_tensor;
  meta_tensor.push_back(ifm_ten);
  meta_tensor.push_back(param_ten);
  meta_tensor.push_back(ofm_ten);

  // ryzenai::xcom::subgraph sg(true);
  // sg.initialize_const_params(params, {});
  // sg.execute(input, output);

  auto xclbin =
      OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu_sg.xclbin";
  auto DPU_KERNEL_NAME = "DPU";

  auto dd_rt = std::make_unique<OpsFusion::FusionSubgraphRuntime>(
      xclbin, DPU_KERNEL_NAME, num_contexts);

  std::vector<uint8_t> txn(txn_bin.size());
  std::memcpy(txn.data(), txn_bin.data(), txn_bin.size());
  std::map<std::string, std::any> attr;
  std::vector<size_t> padded_ifm_tile_offsets = {0, 16 * 32 * 64};

  std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims = {{0, 0, 0},
                                                                  {0, 0, 0}};

  std::vector<std::vector<size_t>> ofm_tile_concat_shapes = {{16, 32, 64},
                                                             {16, 32, 64}};

  std::vector<size_t> ifm_tile_shape = {16, 32, 64};
  std::vector<std::string> in_dtypes = {"int8"};
  std::vector<std::string> out_dtypes = {"int8"};

  attr["mc_code_bin"] = mc_code_bin;
  attr["txn_bin"] = txn;
  attr["num_tiles"] = static_cast<size_t>(2);
  attr["padded_ifm_tile_offsets"] = padded_ifm_tile_offsets;
  attr["padded_ofm_tile_offset_dims"] = padded_ofm_tile_offset_dims;
  attr["ofm_tile_concat_shapes"] = ofm_tile_concat_shapes;
  attr["ifm_tile_shape"] = ifm_tile_shape;
  attr["padded_ifm_tile_shape"] = ifm_tile_shape;
  attr["ofm_tile_shape"] = ifm_tile_shape;
  attr["padded_ofm_tile_shape"] = ifm_tile_shape;
  attr["ifm_addr"] = static_cast<size_t>(0);
  attr["ifm_size"] = static_cast<size_t>(65536);
  attr["param_addr"] = static_cast<size_t>(0);
  attr["param_size"] = static_cast<size_t>(4480);
  attr["ofm_addr"] = static_cast<size_t>(0);
  attr["ofm_size"] = static_cast<size_t>(
      32768); // There is an additional 32 bytes in ddr_range.txt generated by
              // xcompiler. This is manually hardcoded for now
  attr["inter_addr"] = static_cast<size_t>(0);
  attr["inter_size"] = static_cast<size_t>(2097152);
  attr["ofm_orig_tile_size"] = static_cast<size_t>(16 * 32 * 64);
  attr["num_cols_design"] = static_cast<size_t>(4);
  attr["in_dtypes"] = in_dtypes;
  attr["out_dtypes"] = out_dtypes;
  attr["subgraph_name"] = static_cast<std::string>("conv32x32_two_tiles");

  // std::vector<uint8_t> &txn =
  // reinterpret_cast<std::vector<uint8_t>&>(txn_bin);
  dd_rt->init(params, attr);
  for (uint32_t i = 0; i < num_exec_loops; i++) {
    dd_rt->execute_subgraph(input, output);
  }

  int err_count = 0;
  for (int i = 0; i < golden.size(); i++) {
    if (golden.at(i) != ofm.at(i)) {
      std::cout << "golden, actual : " << std::to_string(golden.at(i)) << ", "
                << std::to_string(ofm.at(i)) << "at idx: " << i << std::endl;
      err_count++;
      getchar();
    }
  }

  if (err_count != 0) {
    std::cout << "conv32x32_two_tiles::Test failed with error count : "
              << err_count << std::endl;
    return -1;
  }

  std::cout << "conv32x32_two_tiles::TEST PASS" << std::endl;

  return 0;
}

int conv48x32_three_tiles(uint32_t num_exec_loops, uint32_t num_contexts) {
  std::cout << "Running 48x32 three tile subgraph test" << std::endl;

  auto base_dir = OpInterface::get_dd_base_dir() +
                  "/tests/xcom_subgraphs/0621_conv_48x32_cases/case_tile/";

  auto golden_base_dir =
      OpInterface::get_dd_base_dir() +
      "/tests/xcom_subgraphs/0621_conv_48x32_cases/case_full/";
  auto param = OpsFusion::read_bin_file(base_dir + "param.bin");
  auto ifm = OpsFusion::read_bin_file(golden_base_dir + "ifm.bin");
  auto golden = OpsFusion::read_bin_file(
      golden_base_dir + "golden_Sequential__Sequential_Conv2d_0__ret_fix.bin");
  auto txn_bin = OpsFusion::read_bin_file(base_dir + "mc_code_txn.bin");
  std::vector<std::uint8_t> mc_code_bin =
      OpsFusion::read_bin_file<std::uint8_t>(base_dir + "mc_code.bin");

  std::vector<int8_t> ofm(golden.size(), 0);
  std::vector<Tensor> input;
  std::vector<Tensor> params;
  std::vector<Tensor> output;

  struct Tensor ifm_ten = {ifm.data(), {1, 48, 32, 64}, "int8"};
  input.push_back(ifm_ten);
  struct Tensor param_ten = {param.data(), {1, param.size()}, "int8"};
  params.push_back(param_ten);
  struct Tensor ofm_ten = {ofm.data(), {1, 48, 32, 64}, "int8"};
  output.push_back(ofm_ten);

  std::vector<Tensor> meta_tensor;
  meta_tensor.push_back(ifm_ten);
  meta_tensor.push_back(param_ten);
  meta_tensor.push_back(ofm_ten);

  // ryzenai::xcom::subgraph sg(true);
  // sg.initialize_const_params(params, {});
  // sg.execute(input, output);

  std::vector<uint8_t> txn(txn_bin.size());
  std::memcpy(txn.data(), txn_bin.data(), txn_bin.size());
  std::map<std::string, std::any> attr;

  std::vector<size_t> padded_ifm_tile_offsets = {0, 15 * 32 * 64, 30 * 32 * 64};

  std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims = {
      {0, 0, 0}, {1, 0, 0}, {2, 0, 0}};

  std::vector<std::vector<size_t>> ofm_tile_concat_shapes = {
      {16, 32, 64}, {16, 32, 64}, {16, 32, 64}};

  std::vector<size_t> ifm_tile_shape = {18, 32, 64};
  std::vector<std::string> in_dtypes = {"int8"};
  std::vector<std::string> out_dtypes = {"int8"};

  attr["mc_code_bin"] = mc_code_bin;
  attr["txn_bin"] = txn;
  attr["num_tiles"] = static_cast<size_t>(3);
  attr["padded_ifm_tile_offsets"] = padded_ifm_tile_offsets;
  attr["padded_ofm_tile_offset_dims"] = padded_ofm_tile_offset_dims;
  attr["ofm_tile_concat_shapes"] = ofm_tile_concat_shapes;
  attr["ifm_tile_shape"] = ifm_tile_shape;
  attr["padded_ifm_tile_shape"] = ifm_tile_shape;
  attr["ofm_tile_shape"] = ifm_tile_shape;
  attr["padded_ofm_tile_shape"] = ifm_tile_shape;
  attr["ifm_addr"] = static_cast<size_t>(
      2112); // offset where original, non-tiled ifm has to be copied.
  attr["ifm_size"] =
      static_cast<size_t>(102528); // size required for original non-tiled ifm
  attr["param_addr"] = static_cast<size_t>(0);
  attr["param_size"] = static_cast<size_t>(37248);
  attr["ofm_addr"] = static_cast<size_t>(0);
  attr["ofm_size"] = static_cast<size_t>((18 * 32 * 64));
  attr["inter_addr"] = static_cast<size_t>(0);
  attr["inter_size"] = static_cast<size_t>(2097152);
  attr["ofm_orig_tile_size"] = static_cast<size_t>(16 * 32 * 64);
  attr["num_cols_design"] = static_cast<size_t>(4);
  attr["in_dtypes"] = in_dtypes;
  attr["out_dtypes"] = out_dtypes;
  attr["subgraph_name"] = static_cast<std::string>("conv48x32_three_tiles");

  auto xclbin =
      OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu_sg.xclbin";
  auto DPU_KERNEL_NAME = "DPU";

  auto dd_rt = std::make_unique<OpsFusion::FusionSubgraphRuntime>(
      xclbin, DPU_KERNEL_NAME, num_contexts);

  dd_rt->init(params, attr);
  for (uint32_t i = 0; i < num_exec_loops; i++) {
    dd_rt->execute_subgraph(input, output);
  }

  int err_count = 0;
  for (int i = 0; i < golden.size(); i++) {
    if (golden.at(i) != ofm.at(i)) {
      std::cout << "golden, actual, idx : " << std::to_string(golden.at(i))
                << ", " << std::to_string(ofm.at(i)) << ", " << i << std::endl;
      err_count++;
      getchar();
    }
  }

  if (err_count != 0) {
    std::cout << "conv48x32_three_tiles::Test failed with error count : "
              << err_count << std::endl;
    return -1;
  }

  std::cout << "conv48x32_three_tiles::TEST PASS" << std::endl;

  return 0;
}

int subgraph0_single_tile(uint32_t num_exec_loops) {
  std::cout << "Running 250x250 single tile subgraph test" << std::endl;

  auto base_dir =
      OpInterface::get_dd_base_dir() + "/tests/xcom_subgraphs/case_l2_full/";
  auto param = OpsFusion::read_bin_file(base_dir + "param.bin");
  auto ifm = OpsFusion::read_bin_file(base_dir + "ifm.bin");
  auto golden = OpsFusion::read_bin_file(
      base_dir + "golden_input_108_QuantizeLinear_Output.bin");
  // auto txn_bin = OpsFusion::read_bin_file(base_dir + "mc_code_txn.bin");
  std::vector<std::uint8_t> mc_code_bin =
      OpsFusion::read_bin_file<std::uint8_t>(base_dir + "mc_code.bin");

  std::vector<int8_t> ofm(golden.size(), 0);
  std::vector<Tensor> input;
  std::vector<Tensor> params;
  std::vector<Tensor> output;

  struct Tensor ifm_ten = {ifm.data(), {1, 250, 250, 64}, "int8"};
  input.push_back(ifm_ten);
  struct Tensor param_ten = {param.data(), {1, param.size()}, "int8"};
  params.push_back(param_ten);
  struct Tensor ofm_ten = {ofm.data(), {1, 250, 250, 64}, "int8"};
  output.push_back(ofm_ten);

  std::vector<Tensor> meta_tensor;
  meta_tensor.push_back(ifm_ten);
  meta_tensor.push_back(param_ten);
  meta_tensor.push_back(ofm_ten);

  // std::vector<uint8_t> txn(txn_bin.size());
  // std::memcpy(txn.data(), txn_bin.data(), txn_bin.size());
  std::map<std::string, std::any> attr;
  std::vector<size_t> padded_ifm_tile_offsets = {0};

  std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims = {{0, 0, 0}};

  std::vector<std::vector<size_t>> ofm_tile_concat_shapes = {{250, 250, 64}};

  std::vector<size_t> ifm_tile_shape = {250, 250, 64};
  std::vector<size_t> padded_ifm_tile_shape = ifm_tile_shape;
  std::vector<size_t> ofm_tile_shape = ifm_tile_shape;
  std::vector<size_t> padded_ofm_tile_shape = {252, 256, 64};
  std::vector<std::string> in_dtypes = {"int8"};
  std::vector<std::string> out_dtypes = {"int8"};

  attr["mc_code_bin"] = mc_code_bin;
  // attr["txn_bin"] = txn;
  attr["num_tiles"] = static_cast<size_t>(1);
  attr["padded_ifm_tile_offsets"] = padded_ifm_tile_offsets;
  attr["padded_ofm_tile_offset_dims"] = padded_ofm_tile_offset_dims;
  attr["ofm_tile_concat_shapes"] = ofm_tile_concat_shapes;
  attr["ifm_tile_shape"] = ifm_tile_shape;
  attr["padded_ifm_tile_shape"] = padded_ifm_tile_shape;
  attr["ofm_tile_shape"] = ofm_tile_shape;
  attr["padded_ofm_tile_shape"] = padded_ofm_tile_shape;
  attr["ifm_addr"] = static_cast<size_t>(0);
  attr["ifm_size"] = static_cast<size_t>(4032384);
  attr["param_addr"] = static_cast<size_t>(0);
  attr["param_size"] = static_cast<size_t>(732348);
  attr["ofm_addr"] = static_cast<size_t>(0);
  attr["ofm_size"] = static_cast<size_t>(4128800);
  attr["inter_addr"] = static_cast<size_t>(0);
  attr["inter_size"] = static_cast<size_t>(375171964);
  attr["ofm_orig_tile_size"] = static_cast<size_t>(250 * 250 * 64);
  attr["num_cols_design"] = static_cast<size_t>(4);
  attr["in_dtypes"] = in_dtypes;
  attr["out_dtypes"] = out_dtypes;
  attr["subgraph_name"] =
      static_cast<std::string>("subgraph0_single_tile_250x250");

  auto xclbin =
      OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu_sg.xclbin";
  auto DPU_KERNEL_NAME = "DPU";
  constexpr std::uint32_t num_contexts = 1;

  auto dd_rt = std::make_unique<OpsFusion::FusionSubgraphRuntime>(
      xclbin, DPU_KERNEL_NAME, num_contexts);

  dd_rt->init(params, attr);
  for (uint32_t i = 0; i < num_exec_loops; i++) {
    dd_rt->execute_subgraph(input, output);
  }

  int err_count = 0;
  for (int i = 0; i < golden.size(); i++) {
    if (golden.at(i) != ofm.at(i)) {
      std::cout << "golden, actual : " << std::to_string(golden.at(i)) << ", "
                << std::to_string(ofm.at(i)) << std::endl;
      err_count++;
      getchar();
    }
  }

  if (err_count != 0) {
    std::cout
        << "subgraph0_single_tile_250x250::Test failed with error count : "
        << err_count << std::endl;
    return -1;
  }

  std::cout << "subgraph0_single_tile_250x250::TEST PASS" << std::endl;

  return 0;
}

int subgraph0_two_tile(uint32_t num_exec_loops, uint32_t num_contexts) {
  std::cout << "Running 250x250 two tile subgraph test" << std::endl;

  auto base_dir =
      OpInterface::get_dd_base_dir() + "/tests/xcom_subgraphs/case_l2_tile/";
  auto param = OpsFusion::read_bin_file(base_dir + "param.bin");
  auto ifm = OpsFusion::read_bin_file(base_dir + "ifm.bin");
  auto golden = OpsFusion::read_bin_file(
      base_dir + "golden_input_108_QuantizeLinear_Output.bin");
  // auto txn_bin = OpsFusion::read_bin_file(base_dir + "mc_code_txn.bin");

  std::vector<std::uint8_t> mc_code_bin =
      OpsFusion::read_bin_file<std::uint8_t>(base_dir + "mc_code.bin");

  std::vector<int8_t> ofm(golden.size(), 0);
  std::vector<Tensor> input;
  std::vector<Tensor> params;
  std::vector<Tensor> output;

  struct Tensor ifm_ten = {ifm.data(), {1, 250, 250, 64}, "int8"};
  input.push_back(ifm_ten);
  struct Tensor param_ten = {param.data(), {1, param.size()}, "int8"};
  params.push_back(param_ten);
  struct Tensor ofm_ten = {ofm.data(), {1, 250, 250, 64}, "int8"};
  output.push_back(ofm_ten);

  std::vector<Tensor> meta_tensor;
  meta_tensor.push_back(ifm_ten);
  meta_tensor.push_back(param_ten);
  meta_tensor.push_back(ofm_ten);

  // std::vector<uint8_t> txn(txn_bin.size());
  // std::memcpy(txn.data(), txn_bin.data(), txn_bin.size());
  std::map<std::string, std::any> attr;
  std::vector<size_t> padded_ifm_tile_offsets = {0, 106 * 250 * 64};

  std::vector<std::vector<size_t>> padded_ofm_tile_offset_dims = {{0, 0, 0},
                                                                  {19, 0, 0}};

  std::vector<std::vector<size_t>> ofm_tile_concat_shapes = {{125, 250, 64},
                                                             {125, 250, 64}};

  std::vector<size_t> ifm_tile_shape = {144, 250, 64};
  std::vector<size_t> padded_ifm_tile_shape = ifm_tile_shape;
  std::vector<size_t> ofm_tile_shape = ifm_tile_shape;
  std::vector<size_t> padded_ofm_tile_shape = {144, 256, 64};

  std::vector<std::string> in_dtypes = {"int8"};
  std::vector<std::string> out_dtypes = {"int8"};

  attr["mc_code_bin"] = mc_code_bin;
  // attr["txn_bin"] = txn;
  attr["num_tiles"] = static_cast<size_t>(2);
  attr["padded_ifm_tile_offsets"] = padded_ifm_tile_offsets;
  attr["padded_ofm_tile_offset_dims"] = padded_ofm_tile_offset_dims;
  attr["ofm_tile_concat_shapes"] = ofm_tile_concat_shapes;
  attr["ifm_tile_shape"] = ifm_tile_shape;
  attr["padded_ifm_tile_shape"] = padded_ifm_tile_shape;
  attr["ofm_tile_shape"] = ofm_tile_shape;
  attr["padded_ofm_tile_shape"] = padded_ofm_tile_shape;
  attr["ifm_addr"] = static_cast<size_t>(0);
  attr["ifm_size"] = static_cast<size_t>(4032384);
  attr["param_addr"] = static_cast<size_t>(0);
  attr["param_size"] = static_cast<size_t>(732348);
  attr["ofm_addr"] = static_cast<size_t>(0);
  attr["ofm_size"] = static_cast<size_t>(4128800);
  attr["inter_addr"] = static_cast<size_t>(0);
  attr["inter_size"] = static_cast<size_t>(375171964);
  attr["ofm_orig_tile_size"] = static_cast<size_t>(250 * 250 * 64);
  attr["num_cols_design"] = static_cast<size_t>(4);
  attr["in_dtypes"] = in_dtypes;
  attr["out_dtypes"] = out_dtypes;
  attr["subgraph_name"] =
      static_cast<std::string>("subgraph0_two_tile_250x250");

  auto xclbin =
      OpInterface::get_dd_base_dir() + "/xclbin/stx/4x4_dpu_sg.xclbin";
  auto DPU_KERNEL_NAME = "DPU";

  auto dd_rt = std::make_unique<OpsFusion::FusionSubgraphRuntime>(
      xclbin, DPU_KERNEL_NAME, num_contexts);

  dd_rt->init(params, attr);

  for (uint32_t i = 0; i < num_exec_loops; i++) {
    dd_rt->execute_subgraph(input, output);
  }

  int err_count = 0;
  for (int i = 0; i < golden.size(); i++) {
    if (golden.at(i) != ofm.at(i)) {
      std::cout << "golden, actual : " << std::to_string(golden.at(i)) << ", "
                << std::to_string(ofm.at(i)) << std::endl;
      err_count++;
      getchar();
    }
  }

  if (err_count != 0) {
    std::cout << "subgraph0_two_tile_250x250::Test failed with error count : "
              << err_count << std::endl;
    return -1;
  }

  std::cout << "subgraph0_two_tile_250x250::TEST PASS" << std::endl;

  return 0;
}

int main() {
  conv32x32_single_tile(100);
  conv32x32_two_tiles(100, 1);
  conv32x32_two_tiles(100, 2);

  conv48x32_single_tile(100);
  conv48x32_three_tiles(100, 1);
  conv48x32_three_tiles(100, 2);

  subgraph0_single_tile(100);
  subgraph0_two_tile(100, 1);
  subgraph0_two_tile(100, 2);
}
