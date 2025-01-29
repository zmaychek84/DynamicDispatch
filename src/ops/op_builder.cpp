// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
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

#include <ops/AddTanhLPNorm/AddTanhLPNorm.hpp>
#include <ops/AttentionMaskPrePro/AttentionMaskPrePro.hpp>
#include <ops/DotProductSigmoid/DotProductSigmoid.hpp>
#include <ops/ReduceSum/ReduceSum.hpp>
#include <ops/act_act_matmul_qdq/act_act_matmul_qdq.hpp>
#include <ops/act_const_add/act_const_add.hpp>
#include <ops/act_matmul_softmax/act_matmul_softmax.hpp>
#include <ops/bmm/bmm.hpp>
#include <ops/concat/concat.hpp>
#include <ops/concateOps/concateOps.hpp>
#include <ops/conv/conv.hpp>
#include <ops/conv2matmul/conv2matmul.hpp>
#include <ops/conv2matmul_silu/conv2matmul_silu.hpp>
#include <ops/dequant/dequant.hpp>
#include <ops/dmacompiler/AttentionMaskPrePro_win25/AttentionMaskPrePro_win25.hpp>
#include <ops/dmacompiler/batch_matmul/batch_matmul.hpp>
#include <ops/dmacompiler/gather_qdq_add/gather_qdq_add.hpp>
#include <ops/dmacompiler/matmul_v_geluadd/matmul_v_geluadd.hpp>
#include <ops/dmacompiler/mhapsw/mhapsw.hpp>
#include <ops/dmacompiler/qdq_mul/qdq_mul.hpp>
#include <ops/dmacompiler/qdqadd/qdqadd.hpp>
#include <ops/elwadd/elwadd.hpp>
#include <ops/elwdiv_qdq/elwdiv_qdq.hpp>
#include <ops/elwmul/elwmul.hpp>
#include <ops/elwmul_qdq/elwmul_qdq.hpp>
#include <ops/expand/expand.hpp>
#include <ops/experimental/cube.hpp>
#include <ops/experimental/square.hpp>
#include <ops/flat/mha_v2.hpp>
#include <ops/flat/mlp.hpp>
#include <ops/flat/rms_add.hpp>
#include <ops/gap/gap.hpp>
#include <ops/gelu/gelu.cpp>
#include <ops/gelu_e/gelue.hpp>
#include <ops/groupnorm/groupnorm.hpp>
#include <ops/iconv/iconv.hpp>
#include <ops/l2_norm/l2_norm.hpp>
#include <ops/layernorm/layernorm.hpp>
#include <ops/lstm/lstm.hpp>
#include <ops/maskedsoftmax/maskedsoftmax.hpp>
#include <ops/matmul/matmul.hpp>
#include <ops/matmul_a16a16_mladf/matmul_a16a16_mladf.hpp>
#include <ops/matmul_a16w8_mladf/matmul_a16w8_mladf.hpp>
#include <ops/matmulbias/matmulbias.hpp>
#include <ops/matmulgeluadd/matmulgeluadd.hpp>
#include <ops/matvecadd/matvecadd.hpp>
#include <ops/mha/mha.hpp>
#include <ops/mhachannel/mhachannel.hpp>
#include <ops/mhagprb/mhagprb.hpp>
#include <ops/mhamzdk5/mhamzdk5.hpp>
#include <ops/mhawindow/mhawindow.hpp>
#include <ops/mladfadd/mladfadd.hpp>
#include <ops/mladfelwadd/mladfelwadd.hpp>
#include <ops/mladfelwmul/mladfelwmul.hpp>
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ops/mladfmharope/mladfmharope.hpp>
#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>
#include <ops/mladfsoftmax/mladfsoftmax.hpp>
#include <ops/nni_resize/nni_resize.hpp>
#include <ops/op_builder.hpp>
#include <ops/op_policy.hpp>
#include <ops/pm_load/pm_load.hpp>
#include <ops/preemption/preemption.hpp>
#include <ops/quant/quant.hpp>
#include <ops/record_timer/record_timer.hpp>
#include <ops/sd/concat.hpp>
#include <ops/sd/conv2d.hpp>
#include <ops/sd/elwadd.hpp>
#include <ops/sd/elwmul.hpp>
#include <ops/sd/gelu.hpp>
#include <ops/sd/gemm.hpp>
#include <ops/sd/groupnorm.hpp>
#include <ops/sd/layernorm.hpp>
#include <ops/sd/matmul.hpp>
#include <ops/sd/mha.hpp>
#include <ops/sd/resize.hpp>
#include <ops/sd/silu.hpp>
#include <ops/sd/slice.hpp>
#include <ops/sigmoid/sigmoid.hpp>
#include <ops/silu/silu.hpp>
#include <ops/silu_qdq/silu_qdq.hpp>
#include <ops/slice/slice.hpp>
#include <ops/softmax_qdq/softmax_qdq.hpp>
#include <ops/transpose/transpose.hpp>
#include <ops/unary/unary.hpp>
#include <ops/xcom/conv/conv.hpp>
#include <utils/utils.hpp>

namespace OpsFusion {

bool OpBuilder::is_val_skipped(std::vector<int> pos) {
  return (pos.size() == 0 || std::all_of(pos.begin(), pos.end(),
                                         [](int val) { return val == -1; }));
}

// Safe method to add ops to op registry.

template <typename T, typename... Args>
void OpBuilder::register_op(std::string op_name,
                            const std::vector<std::string> &type_info,
                            const std::vector<int> &pos, Args &&...args)

{

  /*
  Skipping validation for an operation more than once creates ambiguity.
  If validation is skipped during registration for an operation more than
  once, the object will be updated to the latest object.

  Can throw exception ?
  */
  if (OpBuilder::op_registry.count(op_name) == 0) {
    OpBuilder::op_registry[op_name].emplace_back(std::make_unique<T>(
        type_info, pos, op_name, std::forward<Args>(args)...));
    return;
  }
  auto &vec = OpBuilder::op_registry[op_name];
  size_t idx = vec.size() - 1;
  if (is_val_skipped(vec[idx]->get_pos())) {
    if (is_val_skipped(pos)) {
      // throw std::runtime_error("Skipping validation more than once creates
      // ambiguity.")
      vec[idx] = std::make_unique<T>(type_info, pos, op_name,
                                     std::forward<Args>(args)...);
    } else {
      // Making sure that skipped validation object remains
      // at end of the array
      vec.insert(vec.begin() + idx,
                 std::make_unique<T>(type_info, pos, op_name,
                                     std::forward<Args>(args)...));
    }
  } else {
    vec.emplace_back(std::make_unique<T>(type_info, pos, op_name,
                                         std::forward<Args>(args)...));
  }
}

std::unique_ptr<OpInterface>
create_impl(const std::string &op_type, const std::vector<std::string> &types,
            const std::map<std::string, std::any> &attr) {

  // ToDo Deprecate this method. It is not used anywhere now.
  return nullptr;
}

void OpBuilder::init(){
#include "ops/op_registration_db.def"
}

std::unique_ptr<OpInterface> OpBuilder::create(const OpParams &params) {
  REGISTER_KNOWN_OPS(OpBuilder::init)
  auto arg_dtypes = extract_arg_dtypes(params.op_info, params.tensor_map);
  auto &op_name = params.op_name;
  auto &op_type = params.op_info.type;
  try {
    if (op_registry.count(op_type) == 0) {
      std::runtime_error("Provided operator : " + op_name +
                         " is not registerd with OpBuilder.");
    }

    for (const auto &op : op_registry[op_type]) {
      if (op->verify(arg_dtypes)) {
        return op->create(params);
      }
    }
    throw std::runtime_error(
        "Provided Datatypes are not supported by current " + op_name +
        " Impl.");
  } catch (std::exception &e) {
    std::ostringstream oss;
    for (int i = 0; i < arg_dtypes.size(); ++i) {
      oss << i << ":" << arg_dtypes.at(i) << " ";
    }
    DD_THROW(dd_format("OpBuilder::create() failed.\n"
                       "Details:\n"
                       "  OpName: {}\n"
                       "  OpType: {}\n"
                       "  Provided arg dtypes: {}\n"
                       "  Error: {}",
                       op_name, params.op_info.type, oss.str(), e.what()));

    // return nullptr to fix compiler warning. Control should not reach here.
    return nullptr;
  }
}

std::unique_ptr<OpInterface> OpBuilder::create(
    const std::string &op_name, const Metadata::OpInfo &op_info,
    const std::map<std::string, Metadata::OffsetInfo> &tensor_map) {
  auto params = OpParams(op_name, op_info, tensor_map);
  return create(params);
}

bool OpBuilder::is_supported(const std::string &op_type,
                             const std::vector<std::string> &types,
                             const std::map<std::string, std::any> &attr) {
  for (const auto &op : op_registry[op_type]) {
    if (op->verify(types)) {
      return true;
    }
  }
  return false;
}

} // namespace OpsFusion
