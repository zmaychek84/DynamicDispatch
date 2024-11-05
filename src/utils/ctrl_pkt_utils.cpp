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

#include <nlohmann/json.hpp>
#include <string>

#include "ctrl_pkt_utils.hpp"
#include "utils/tfuncs.hpp"

using json = nlohmann::json;

std::vector<CtrlPktPatchInfo>
json_str_to_ctrlpkt_patch_info(const std::vector<uint8_t> &json_vec) {
  std::string json_str(json_vec.begin(), json_vec.end());
  json data;
  try {
    data = json::parse(json_str, nullptr, true);
  } catch (std::exception &e) {
    DD_THROW(OpsFusion::dd_format(
        "Failed to parse ctrl pkt meta JSON String: (Detail: {})", e.what()));
  }

  RYZENAI_LOG_TRACE("Loading the ctrl_pkt_meta.json ... DONE");
  std::vector<CtrlPktPatchInfo> patch_info;
  try {
    for (const auto &pi : data.at("ctrl_pkt_patch_info")) {
      patch_info.push_back({pi.at("offset"), pi.at("size"),
                            pi.at("xrt_arg_idx"), pi.at("bo_offset")});
    }
  } catch (std::exception &e) {
    DD_THROW(OpsFusion::dd_format(
        "Failed to parse ctrl pkt meta JSON String: (Detail: {})", e.what()));
  }

  return patch_info;
}
