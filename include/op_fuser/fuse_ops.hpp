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

#pragma once

#include <any>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

#include "fuse_types.hpp"
#include <ops/op_builder.hpp>
#include <ops/op_interface.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>

#include "txn/txn_utils.hpp"

using json = nlohmann::json;

namespace OpsFusion {

static std::ostream &operator<<(std::ostream &os, const Metadata &m) {
  for (const auto &opinfo : m.op_list) {
    os << opinfo.name << " : " << opinfo.type << " : ";
    for (const auto &arg : opinfo.in_args) {
      os << "  in: " << arg << '\n';
    }
    for (const auto &arg : opinfo.const_args) {
      os << "  const: " << arg << '\n';
    }
    for (const auto &arg : opinfo.out_args) {
      os << "  out: " << arg << '\n';
    }
    os << std::endl;
  }

  for (const auto &[name, tinfo] : m.fused_tensors) {
    os << name << " : " << tinfo.size << ", " << tinfo.arg_idx << std::endl;
  }

  for (const auto &[name, off_info] : m.tensor_map) {
    os << name << " : " << off_info.parent_name << ", " << off_info.offset
       << ", " << off_info.arg_idx << std::endl;
  }

  for (const auto &[name, span_info] : m.super_instr_map) {
    os << "Super Kernel Instr Span : " << name << " : " << span_info.offset
       << ", " << span_info.size << std::endl;
  }
  for (const auto &[key, span_info] : m.const_map) {
    os << "const Span : " << key << " - " << span_info.offset << ", "
       << span_info.size << std::endl;
  }

  return os;
}

using txn_vec_t = std::vector<uint8_t>;

template <typename T>
static T json_get(const json &js, const std::string &key, const T &value) {
  return js.find(key) != js.end() ? js.at(key).template get<T>() : value;
}

static std::map<std::string, std::any> extract_op_attrs(const json &op_info) {
  std::map<std::string, std::any> attrs;
  if (op_info.find("attrs") == op_info.end()) {
    return attrs;
  }

  for (const auto &[attr_name, attr_info] : op_info.at("attrs").items()) {
    const std::string dtype = attr_info.at("type").template get<std::string>();
    const std::vector<std::string> values =
        attr_info.at("value").template get<std::vector<std::string>>();

    if (dtype == "float") {
      attrs[attr_name] =
          for_each(values, [](const auto &s) { return std::stof(s); });
    } else if (dtype == "int") {
      attrs[attr_name] =
          for_each(values, [](const auto &s) { return std::stoi(s); });
    } else if (dtype == "str") {
      attrs[attr_name] = values;
    } else {
      DD_THROW(OpsFusion::dd_format("Unsupported dtype for attrs in JSON: {}",
                                    dtype));
    }
  }
  return attrs;
}

static std::map<std::string, std::any> load_aux_info(const json &aux_info) {
  std::map<std::string, std::any> res;

  // Original outputs
  {
    if (aux_info.find("original_outputs") != aux_info.end()) {
      std::map<std::string, Tensor> tensors;
      for (const auto &[name, tinfo] :
           aux_info.at("original_outputs").items()) {
        Tensor tensor{nullptr,
                      tinfo.at("shape").template get<std::vector<size_t>>(),
                      tinfo.at("dtype").template get<std::string>()};
        tensors[name] = tensor;
      }
      res["original_outputs"] = std::any(tensors);
    }
  }

  // Original Inputs
  {
    if (aux_info.find("original_inputs") != aux_info.end()) {
      std::map<std::string, Tensor> tensors;
      for (const auto &[name, tinfo] : aux_info.at("original_inputs").items()) {
        Tensor tensor{nullptr,
                      tinfo.at("shape").template get<std::vector<size_t>>(),
                      tinfo.at("dtype").template get<std::string>()};
        tensors[name] = tensor;
      }
      res["original_inputs"] = std::any(tensors);
    }
  }

  return res;
}

/// Load Metadata from in-memory string
static Metadata load_meta_string(const std::string &meta_string) {
  // TODO : Nothing is caught while parse error
  json data;
  try {
    data = json::parse(meta_string, nullptr, true);
  } catch (std::exception &e) {
    DD_THROW(OpsFusion::dd_format("Failed to parse JSON String: (Detail: {})",
                                  e.what()));
  }
  RYZENAI_LOG_TRACE("Loading the meta.json ... DONE");

  Metadata meta;
  meta.json_path = "in-memory string";
  meta.major_version = data.at("dd_meta_major_version");
  meta.minor_version = data.at("dd_meta_minor_version");
  std::string ver = std::to_string(meta.major_version) + "." +
                    std::to_string(meta.minor_version);
  RYZENAI_LOG_TRACE("DD Meta Version: " + ver);
  // oplist
  for (const auto &opinfo : data.at("op_list")) {
    meta.op_list.push_back(
        {opinfo.at("name").template get<std::string>(),
         opinfo.at("type").template get<std::string>(),
         opinfo.at("in_args").template get<std::vector<std::string>>(),
         opinfo.at("const_args").template get<std::vector<std::string>>(),
         opinfo.at("out_args").template get<std::vector<std::string>>(),
         {}});
    meta.op_list.back().attr = extract_op_attrs(opinfo);
  }

  // tensor info
  for (const auto &[name, tinfo] : data.at("fused_tensors").items()) {
    meta.fused_tensors[name] = {
        tinfo.at("buffer_size").template get<size_t>(),
        tinfo.at("xrt_arg_id").template get<size_t>(),
        tinfo.at("packed_tensors").template get<std::vector<std::string>>()};
  }

  // tensor_map
  for (const auto &[name, offset_info] : data.at("tensor_map").items()) {
    meta.tensor_map[name] = {
        offset_info.at("packed_buffer_label").template get<std::string>(),
        offset_info.at("offset").template get<size_t>(),
        offset_info.at("xrt_arg_id").template get<size_t>(),
        offset_info.at("dtype").template get<std::string>(),
        offset_info.at("shape").template get<std::vector<size_t>>(),
        offset_info.at("size_in_bytes").template get<size_t>(),
        json_get<std::string>(offset_info, "file_name", ""),
        json_get<size_t>(offset_info, "file_size", 0)};
  }

  if (data.find("aux_info") != data.end()) {
    meta.aux_info = load_aux_info(data.at("aux_info"));
  }

  RYZENAI_LOG_TRACE("Filling Metadata ... DONE");
  return meta;
}

static Metadata load_meta_json(const std::string &meta_json) {
  RYZENAI_LOG_TRACE("Loading the meta.json ...");
  std::ifstream ifs(meta_json);
  DD_ASSERT(ifs.is_open(),
            OpsFusion::dd_format("Couldn't open JSON : {}", meta_json));
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string meta_string = ss.str();
  auto meta = load_meta_string(meta_string);
  meta.json_path = meta_json;
  return meta;
}

static Metadata load_meta_json_str(const std::string &meta_json_str) {
  auto meta = load_meta_string(meta_json_str);
  meta.json_path = "in-memory string";
  return meta;
}

} // namespace OpsFusion
