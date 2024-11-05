// Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

#include "op_fuser/fuse_ops.hpp"
#include "op_fuser/fusion_rt.hpp"
#include "ops/op_builder.hpp"
#include "ops/ops_common/weight_bias_f2bfp.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

// Need derived class for passing numpy / pytorch tensors
class FusionRuntime : public OpsFusion::FusionRuntime {
public:
  // Constructor just forwards args to base class
  FusionRuntime() : OpsFusion::FusionRuntime() {}
  FusionRuntime(const std::string &xclbin) : OpsFusion::FusionRuntime(xclbin) {}

  void execute_ndarrays(
      const std::vector<nb::ndarray<nb::c_contig>> &input_ndarrays,
      const std::vector<nb::ndarray<nb::c_contig>> &output_ndarrays) {

    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;

    for (const auto &input_ndarray : input_ndarrays) {
      inputs.push_back(ndarray_to_tensor(input_ndarray));
    }

    for (const auto &output_ndarray : output_ndarrays) {
      outputs.push_back(ndarray_to_tensor(output_ndarray));
    }

    // Base class accepts Tensor structs
    execute(inputs, outputs);
  }

  // FIX ME: temporary work around till DDConfig structure can be exposed to
  // python directly.
  void compile_dd(const OpsFusion::Metadata &meta, std::string xclbin_name) {
    OpsFusion::DDConfig cfg;
    auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_name);
    cfg.xclbin_content = &xclbin_content;
    compile(meta, "", cfg);
  }

private:
  // Utility function to convert ndarray to Tensor
  static Tensor ndarray_to_tensor(const nb::ndarray<nb::c_contig> &ndarray) {
    void *data = const_cast<void *>(
        ndarray.data()); // Tensor data is not defined as const
    std::vector<size_t> shape;
    std::string dtype;

    for (size_t i = 0; i < ndarray.ndim(); ++i) {
      shape.push_back(ndarray.shape(i));
    }

    if (ndarray.dtype() == nb::dtype<int16_t>()) {
      dtype = "bfloat16";
    } else if (ndarray.dtype() == nb::dtype<uint8_t>()) {
      dtype = "uint8";
    } else if (ndarray.dtype() == nb::dtype<int8_t>()) {
      dtype = "int8";
    } else if (ndarray.dtype() == nb::dtype<float>()) {
      dtype = "float";
    } else {
      throw std::runtime_error("Unsupported data type");
    }

    return Tensor{data, shape, dtype};
  }
};

NB_MODULE(_DynamicDispatch, m) {
  nb::class_<OpsFusion::Metadata> metadata(m, "Metadata");

  nb::class_<OpsFusion::Metadata::OpInfo>(metadata, "OpInfo")
      .def(nb::init<std::string, std::string, std::vector<std::string>>())
      .def_rw("name", &OpsFusion::Metadata::OpInfo::name)
      .def_rw("type", &OpsFusion::Metadata::OpInfo::type)
      .def_rw("in_args", &OpsFusion::Metadata::OpInfo::in_args)
      .def_rw("const_args", &OpsFusion::Metadata::OpInfo::const_args)
      .def_rw("out_args", &OpsFusion::Metadata::OpInfo::out_args);

  nb::class_<OpsFusion::Metadata::TensorInfo>(metadata, "TensorInfo")
      .def_rw("size", &OpsFusion::Metadata::TensorInfo::size)
      .def_rw("arg_idx", &OpsFusion::Metadata::TensorInfo::arg_idx)
      .def_rw("packed_tensors",
              &OpsFusion::Metadata::TensorInfo::packed_tensors);

  nb::class_<OpsFusion::Metadata::OffsetInfo>(metadata, "OffsetInfo")
      .def_rw("parent_name", &OpsFusion::Metadata::OffsetInfo::parent_name)
      .def_rw("offset", &OpsFusion::Metadata::OffsetInfo::offset)
      .def_rw("arg_idx", &OpsFusion::Metadata::OffsetInfo::arg_idx)
      .def_rw("dtype", &OpsFusion::Metadata::OffsetInfo::dtype)
      .def_rw("shape", &OpsFusion::Metadata::OffsetInfo::shape)
      .def_rw("size_in_bytes", &OpsFusion::Metadata::OffsetInfo::size_in_bytes)
      .def_rw("file_name", &OpsFusion::Metadata::OffsetInfo::file_name)
      .def_rw("file_size", &OpsFusion::Metadata::OffsetInfo::file_size);

  nb::class_<OpsFusion::Metadata::Span>(metadata, "Span")
      .def_rw("offset", &OpsFusion::Metadata::Span::offset)
      .def_rw("size", &OpsFusion::Metadata::Span::size);

  metadata.def_rw("op_list", &OpsFusion::Metadata::op_list);
  metadata.def_rw("fused_tensors", &OpsFusion::Metadata::fused_tensors);
  metadata.def_rw("tensor_map", &OpsFusion::Metadata::tensor_map);
  metadata.def_rw("super_instr_map", &OpsFusion::Metadata::super_instr_map);
  metadata.def_rw("const_map", &OpsFusion::Metadata::const_map);

  nb::class_<OpsFusion::OpBuilder> opBuilder(m, "OpBuilder");
  opBuilder.def_static("is_supported", &OpsFusion::OpBuilder::is_supported);

  m.def("load_meta_json", OpsFusion::load_meta_json);

  nb::class_<FusionRuntime>(m, "FusionRuntime")
      .def(nb::init<const std::string &>())
      .def(nb::init<>())
      .def(
          "compile",
          [](FusionRuntime &self, const OpsFusion::Metadata &meta,
             std::string xclbin_name) { self.compile_dd(meta, xclbin_name); },
          "Function to configure the fusion runtime and compile metadata.")
      .def(
          "save_state",
          [](FusionRuntime &self, std::string &metaname) {
            self.save_state(metaname);
          },
          "Function to save fusion runtime metastate.")
      .def(
          "load_state",
          [](FusionRuntime &self, std::string &metaname) {
            self.load_state(metaname);
          },
          "Function to load the fusion runtime metastate.")
      .def(
          "init",
          [](FusionRuntime &self, const OpsFusion::Metadata &meta) {
            self.init(meta);
          },
          "Function to configure the fusion runtime.")
      .def("execute", &FusionRuntime::execute_ndarrays,
           "Function to initiate inference.");

  auto bfp16 = m.def_submodule("bfp16", "bfp16 submodule");
  bfp16.def("const_from_fp32_to_bfp16", &const_from_fp32_to_bfp16,
            "Convert weight and bias from float(2D) to 1D bfp16 array");
}
