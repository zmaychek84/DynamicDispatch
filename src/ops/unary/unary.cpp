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

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>
// #include <utils/dpu_kernel_metadata.hpp>

#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>
#include <vector>
#include <xrt_context/xrt_context.hpp>

#include <ops/op_interface.hpp>
#include <ops/unary/unary.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

namespace ryzenai {

template <typename InT, typename OutT>
void unary<InT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::tuple<int, int>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    std::tuple<int, int> mat = supported_shapes[i];
    std::string key =
        get_instr_key(txn_fname_prefix_, std::get<0>(mat), std::get<1>(mat));
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}

template <typename InT, typename OutT>
const std::vector<uint8_t> unary<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K] = extract_MK(input);
  std::string txn_key = get_instr_key(txn_fname_prefix_, M, K);

  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);

  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}
template <typename InT, typename OutT> void unary<InT, OutT>::look_up_bins() {
  // std::cout <<  get_name() << " look_up_bin"<< '\n';
  Transaction &txn = Transaction::getInstance();

  std::string fsep = "/";
  std::string attsep = "_";
  std::string ext = ".xclbin";

  //(.+[/\\])(([a-z]+_)+)([0-9]{1}[xX]{1}[0-9]{1})_

  std::vector<std::string> bins;
  std::regex pattern_class(get_name());

  //* Because txn.transactionlist is not accepted and we like to add
  // manually all dimensions and bins this code cannot be compiled
  // or executed */
  /* This code is ommitted because we cannot commit otherwise
  std::cout << get_name() << " look_up_bin"<<  txn.transactionlist.size() <<
  '\n'; for (const std::string &entry : txn.transactionlist) { std::smatch
  match;
    // std::cout <<  get_name() << " look_up_bin"<< entry << '\n';
    if (std::regex_search(entry.begin(), entry.end(), match, pattern_class)) {
      bins.push_back(entry);
    }
  }
  */

  // the naming convention :
  // file_system + / name _ arch _ operands attributes _ sizes _ extras .[bin,
  // xclbin] name = letter+ ( _ name )* arch = N x N operand attributes =
  // [letter + type]+ type = [int, int8, int16, bf16] sizes = number+ (_
  // sizes)* extras = letter

  std::string f_regex = "([.]+/)";
  std::string arch_regex = "(([0-9]+[xX])*[0-9]+_)*";
  std::string name_regex = "(([a-z]+_)+)"; // std::regex::icase);
  std::string op_regex =
      "([abc](int|bfloat|float)*[0-9]+)+_"; // std::regex::icase);
  std::string sh_regex = "(([0-9]+_)+[0-9\\.]*)";
  std::string ex_regex = "(.+bin)";
  std::string end_regex = "*bin";

  // std::cout << f_regex+name_regex+arch_regex << std::endl;
  //                        1+2    3+4         5+6      7 +8      9
  std::regex pattern(name_regex + arch_regex + op_regex +
                         sh_regex, //+ex_regex,//+end_regex,
                                   // f_regex, //+
                                   //+arch_regex,
                                   //+op_regex+sh_regex+ex_regex,
                     std::regex::icase);

  std::vector<std::vector<int>> Sdescriptions;

  for (const std::string &i : bins) {
    std::string binname, name, arch, op, shapes;
    std::smatch match;
    // std::cout << "i " << i << std::endl;
    if (std::regex_search(i.begin(), i.end(), match, pattern)) {
      // for (std::string  m : match)
      //   std::cout << "  submatch " << m << '\n';

      binname = match[0];
      if (binname.find(name) < 0) {
        throw std::runtime_error(name +
                                 "IPU Wrapper expect to have transaction "
                                 "bine named after the class.");
      }

      name = match[1];
      arch = match[3];
      if (arch.size() == 0) {
        arch = "4x4_";
      } // By default
      op = match[5];
      if (op.size() == 0) {
        op = "abfloat16cbfloat16_";
      } // By default
      shapes = match[7];
      if (shapes.size() > 0) {
        // std::cout << "S " << shapes << "\n";
        std::vector<int> res = std::vector<int>();
        std::string item, t;
        int t1 = 0;
        size_t posl = 0, posr = shapes.find("_", posl);
        item = shapes.substr(posl, posr);
        // std::cout << "I " << item << "\n";
        size_t M = shapes.size();
        while (posl < M) {
          t1 += 1;
          if (t1 > 4) {
            throw std::runtime_error(name + "Paolo you idiot ");
          }
          res.push_back(stoi(item));
          posl = posr + 1;
          posr = shapes.find("_", posl);
          posr = (posr < 0 || posr > M) ? M - 1 : posr;
          item = shapes.substr(posl, posr);
          // std::cout << "I " << item << " " << posl << " "<< posr << "\n";
        }
        // for (int i=0; i<res.size(); i++ )
        //   std::cout << "Res " << res[i] << "\n";

        if (default_shapes_.count(name + arch + op) == 0) {
          default_shapes_[name + arch + op] =
              std::vector<std::tuple<int, int>>();
        }
        default_shapes_[name + arch + op].push_back(extract_MK_(res));
      }
    }
  }
}
template <typename InT, typename OutT>
void unary<InT, OutT>::build(const std::string &operand_dtype, bool load_xrt) {

  if (operand_dtype != "bfloat16") {
    std::runtime_error("Unary only supports bfloat16 data type "
                       "for operand and result");
  }
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(InT);

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = get_xclbinname();

  std::string arch_regex = "(([0-9]+[xX])*[0-9]+_)*";
  std::regex pattern(arch_regex);
  std::smatch match;
  std::string arch = "";
  if (std::regex_search(XCLBIN_FNAME, match, pattern)) {
    arch = match[0];
    if (arch.size() > 0) {
      arch += "_";
    } else {
      arch = "4x4_";
    }
  }

  txn_fname_prefix_ = get_name() + "_" + arch + "a" +
                      txnbin_operand_header.at(operand_dtype) + "c" +
                      txnbin_operand_header.at(operand_dtype);

  //  populate the default shapes by looking up or by default
  if (0) {
    look_up_bins();
  } else {
    populate_default_shapes(operand_dtype);
  }

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });

    // preempting bo creation with largest shape for unit testing
    const size_t bo_size_in_bytes =
        max_pairwise_product(
            default_shapes_[get_name() +
                            txnbin_operand_header[operand_dtype]]) *
        operand_dtype_size_;

    a_bo_ = xrt::bo(xrt_ctx_->get_device(), bo_size_in_bytes,
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    c_bo_ = xrt::bo(xrt_ctx_->get_device(), bo_size_in_bytes,
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = " Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[" + get_name() +
                    "] ID: " + std::to_string(0) + // unary_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    operand_dtype + ", " + operand_dtype + ")");
}

template <typename InT, typename OutT>
void unary<InT, OutT>::execute(std::vector<Tensor> &input,
                               std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("unary IPU Wrapper expect to have one input.");
  }
  const int a_idx = 0;
  // The first data is a and second data is b
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;

  if (!isSupportedShape(input.at(a_idx))) {
    std::runtime_error("Unsupported matrix dimensions for unary");
  }

  double exec_start = GET_ELAPSED_TIME_NS();

  const auto operand_size_in_bytes =
      input.at(a_idx).shape.at(0) * input.at(a_idx).shape.at(1) * sizeof(InT);
  RYZENAI_LOG_TRACE("elwmul: operand_size_in_bytes:" +
                    std::to_string(operand_size_in_bytes));

  // TODO not really sure we need this member
  operand_shape_[0] = input.at(0).shape.at(0);
  operand_shape_[1] = input.at(0).shape.at(1);

  const auto bo_size_in_bytes =
      input.at(0).shape.at(0) * input.at(0).shape.at(1) * operand_dtype_size_;
  /* Create input/output BOs */
  RYZENAI_LOG_TRACE("elwmul: A_BO_SIZE:" + std::to_string(bo_size_in_bytes) +
                    " C_BO_SIZE:" + std::to_string(bo_size_in_bytes));

  // a_bo copy
  double a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes);
  double a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  double a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  double a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  std::vector<xrt::bo> inputs = {a_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};

  double run_aie_start = GET_ELAPSED_TIME_NS();
  set_kernel_shape(input.at(0).shape);
  execute(inputs, outputs);

  double run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;

  // sync output activation to host memory
  double c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  double c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;

  // copy c_bo to host memory
  auto aie_out = (OutT *)output.at(0).data;
  double c_copy_start = GET_ELAPSED_TIME_NS();
  OutT *c_bo_map = c_bo_.map<OutT *>();
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes);
  double c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  double exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      // std::to_string(unary_id_) + " " + std::to_string(operand_shape_[0]) +
      get_name() + " " + std::to_string(operand_shape_[0]) + " " +
      std::to_string(operand_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template class unary<uint16_t, uint16_t>;

} // namespace ryzenai
