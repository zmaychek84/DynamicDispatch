#include <filesystem>
#include <fstream>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <stdio.h>
#include <utils/tfuncs.hpp>
#include <utils/tmpfile.hpp>

#include "metastate_api.hpp"

namespace gpb = google::protobuf;
namespace fs = std::filesystem;

/// sz : Size in Bytes
template <typename T>
static void set_buffer(dd_proto::Buffer *buffer, const T *data, size_t sz) {
  buffer->set_data(data, sz);
  buffer->set_size(sz);
  buffer->mutable_hash()->set_type("STL");
  buffer->mutable_hash()->set_value("00001111");
}

template <typename T, typename Msg>
static gpb::Any stdanyvec_to_pbany(const std::any &stdany) {
  Msg vec;
  const auto &data = std::any_cast<const std::vector<T> &>(stdany);
  vec.mutable_data()->Add(data.begin(), data.end());

  gpb::Any pbany;
  pbany.PackFrom(vec);
  return pbany;
}

template <typename T, typename Msg>
static std::any pbany_to_stdanyvec(const gpb::Any &pbany) {
  Msg proto_vec;
  DD_ASSERT(pbany.UnpackTo(&proto_vec),
            OpsFusion::dd_format("Protobuf Any unpack failed"));
  std::vector<T> v(proto_vec.data().begin(), proto_vec.data().end());
  std::any stdany = v;
  return stdany;
}

template <typename T, typename Msg>
static gpb::Any stdany_to_pbany(const std::any &stdany) {
  const auto &data = std::any_cast<const T &>(stdany);

  Msg d;
  d.set_data(data);
  gpb::Any pbany;
  pbany.PackFrom(d);
  return pbany;
}

template <typename T, typename Msg>
static std::any pbany_to_stdany(const gpb::Any &pbany) {
  Msg proto_vec;
  DD_ASSERT(pbany.UnpackTo(&proto_vec),
            OpsFusion::dd_format("Protobuf Any unpack failed"));
  // std::vector<T> v(proto_vec.data().begin(), proto_vec.data().end());
  T v = proto_vec.data();
  std::any stdany = v;
  return stdany;
}

namespace OpsFusion {
MetaStateAPI::MetaStateAPI()
    : metastate_(std::make_shared<dd_proto::MetaState>()) {}

/// sz : Size in Bytes
void MetaStateAPI::set_buffer_file(dd_proto::BinFile *binfile, FILE *file,
                                   const std::string &dir,
                                   const std::string &filename) const {
  fseek64(file, 0, SEEK_END);
  auto sz = ftell64(file);
  fseek64(file, 0, SEEK_SET);
  if (!save_func_) {
    auto filepath = fs::path{dir} / filename;
    Utils::save_tmpfile_on_disk(filepath,
                                file); // a memory efficient implementation
  } else {
    save_func_(filename, file);
  }
  binfile->set_file(filename);
  binfile->set_size(sz);
  binfile->mutable_hash()->set_type("STL");
  binfile->mutable_hash()->set_value("00001111");
}

FILE *MetaStateAPI::get_buffer_file(const dd_proto::BinFile &binfile,
                                    const std::string &dir) const {
  const auto &file_name = binfile.file();
  FILE *ret;
  if (load_func_) {
    ret = load_func_((fs::path{dir} / file_name).string());
  } else {
#ifdef _WIN32
    fopen_s(&ret, (fs::path{dir} / file_name).string().c_str(), "rb");
#else
    ret = fopen((fs::path{dir} / file_name).string().c_str(), "rb");
#endif
  }
  fseek64(ret, 0, SEEK_END);
  auto size = ftell64(ret);
  fseek64(ret, 0, SEEK_SET);
  DD_ASSERT(binfile.size() == size,
            "Size of const_bo data read from file doesn't match with size "
            "specified in metastate");
  return ret;
}

MetaStateAPI &MetaStateAPI::update_meta(const Metadata &meta) {

  metastate_->set_major_version(meta.major_version);
  metastate_->set_minor_version(meta.minor_version);

  // Op List
  for (const auto &op : meta.op_list) {
    auto *proto_op = metastate_->add_op_list();
    proto_op->set_name(op.name);
    proto_op->set_type(op.type);
    proto_op->set_pdi_id(op.pdi_id);

    proto_op->mutable_in_args()->Add(op.in_args.begin(), op.in_args.end());
    proto_op->mutable_const_args()->Add(op.const_args.begin(),
                                        op.const_args.end());
    proto_op->mutable_out_args()->Add(op.out_args.begin(), op.out_args.end());

    set_buffer(proto_op->mutable_txn_bin(), op.txn_bin.data(),
               op.txn_bin.size());

    for (const auto &patch_info : op.ctrl_pkt_patch_info) {
      auto *proto_ctrl_pkt_patch_info = proto_op->add_ctrl_pkt_patch_info();
      proto_ctrl_pkt_patch_info->set_size(patch_info.size);
      proto_ctrl_pkt_patch_info->set_offset(patch_info.offset);
      proto_ctrl_pkt_patch_info->set_xrt_arg_idx(patch_info.xrt_arg_idx);
      proto_ctrl_pkt_patch_info->set_bo_offset(patch_info.bo_offset);
    }

    // Attrs
    for (const auto &[name, attr] : op.attr) {
      if (attr.type() == typeid(std::vector<int>)) {
        (*proto_op->mutable_attrs())[name] =
            stdanyvec_to_pbany<int, dd_proto::IntVector>(attr);
      } else if (attr.type() == typeid(std::vector<float>)) {
        (*proto_op->mutable_attrs())[name] =
            stdanyvec_to_pbany<float, dd_proto::FloatVector>(attr);
      } else if (attr.type() == typeid(std::vector<std::string>)) {
        (*proto_op->mutable_attrs())[name] =
            stdanyvec_to_pbany<std::string, dd_proto::StringVector>(attr);
      } else if (attr.type() == typeid(unsigned int)) {
        (*proto_op->mutable_attrs())[name] =
            stdany_to_pbany<uint32_t, dd_proto::UInt32>(attr);
      } else if (attr.type() == typeid(std::string)) {
        (*proto_op->mutable_attrs())[name] =
            stdany_to_pbany<std::string, dd_proto::String>(attr);
      } else {
        // std::cout << (attr.type()) << std::endl;
        std::cout << "Op type: " << op.type << std::endl;
        std::cout << "Op Name: " << op.name << std::endl;
        DD_THROW("Unsupported DataType in Attrs: "s + attr.type().name());
      }
    }
  }

  // Fused Tensors
  auto &proto_fused_tensors = *metastate_->mutable_fused_tensors();
  for (const auto &[name, tinfo] : meta.fused_tensors) {
    auto &proto_tinfo = proto_fused_tensors[name];
    proto_tinfo.set_size(tinfo.size);
    proto_tinfo.set_xrt_arg_idx(tinfo.arg_idx);
    proto_tinfo.mutable_packed_tensors()->Add(tinfo.packed_tensors.begin(),
                                              tinfo.packed_tensors.end());
  }

  // Tensor Map
  auto &proto_tensor_map = *metastate_->mutable_tensor_map();
  for (const auto &[name, tinfo] : meta.tensor_map) {
    auto &proto_tinfo = proto_tensor_map[name];
    proto_tinfo.set_parent_name(tinfo.parent_name);
    proto_tinfo.set_offset(tinfo.offset);
    proto_tinfo.set_xrt_arg_idx(tinfo.arg_idx);
    proto_tinfo.set_dtype(tinfo.dtype);
    proto_tinfo.mutable_shape()->Add(tinfo.shape.begin(), tinfo.shape.end());
    proto_tinfo.set_size_in_bytes(tinfo.size_in_bytes);
    if (!tinfo.file_name.empty()) {
      proto_tinfo.set_file_name(tinfo.file_name);
      proto_tinfo.set_file_size(tinfo.file_size);
    }
  }

  // Super Instr Map
  auto &proto_super_map = *metastate_->mutable_super_instr_map();
  for (const auto &[name, tspan] : meta.super_instr_map) {
    auto &proto_span = proto_super_map[name];
    proto_span.set_offset(tspan.offset);
    proto_span.set_size(tspan.size);
  }

  // ConstMap
  auto &proto_const_map = *metastate_->mutable_const_map();
  for (const auto &[name, tspan] : meta.const_map) {
    auto &proto_span = proto_const_map[name];
    proto_span.set_offset(tspan.offset);
    proto_span.set_size(tspan.size);
  }

  // ctrl_pkt_map
  auto &proto_ctrl_pkt_map = *metastate_->mutable_control_pkt_map();
  for (const auto &[name, tspan] : meta.ctrl_pkt_map) {
    auto &proto_span = proto_ctrl_pkt_map[name];
    proto_span.set_offset(tspan.offset);
    proto_span.set_size(tspan.size);
  }

  metastate_->mutable_scratch_op_set()->Add(meta.scratch_op_set.begin(),
                                            meta.scratch_op_set.end());
  metastate_->set_max_op_scratch_pad_size(meta.max_op_scratch_pad_size);
  metastate_->set_max_tensor_padding_sz(meta.max_tensor_padding_sz);
  metastate_->set_json_path(meta.json_path);

  // Partition
  for (const auto &tpartition : meta.partitions) {
    auto *proto_part = metastate_->add_partitions();
    proto_part->set_start_idx(tpartition.op_range.first);
    proto_part->set_end_idx(tpartition.op_range.second);
    proto_part->set_pdi_id(tpartition.pdi_id);
  }

  return *this;
}

MetaStateAPI &MetaStateAPI::update_save_func(save_function func) {
  save_func_ = func;

  return *this;
}

MetaStateAPI &MetaStateAPI::update_const_bo(FILE *const_bo,
                                            const std::string &cache_dir,
                                            const std::string &filename) {
  set_buffer_file(metastate_->mutable_const_bo(), const_bo, cache_dir,
                  filename);
  return *this;
}

MetaStateAPI &MetaStateAPI::update_superinstr_bo(FILE *superinstr_bo,
                                                 const std::string &cache_dir,
                                                 const std::string &filename) {
  set_buffer_file(metastate_->mutable_superinstr_bo(), superinstr_bo, cache_dir,
                  filename);
  return *this;
}

MetaStateAPI &MetaStateAPI::update_input_bo(FILE *input_bo,
                                            const std::string &cache_dir,
                                            const std::string &filename) {
  set_buffer_file(metastate_->mutable_input_bo(), input_bo, cache_dir,
                  filename);
  return *this;
}

MetaStateAPI &MetaStateAPI::update_ctrl_pkt_bo(FILE *ctrl_pkt_bo,
                                               const std::string &cache_dir,
                                               const std::string &filename) {
  if ((metastate_->major_version() >= 1) &&
      (metastate_->minor_version() >= 1)) {
    set_buffer_file(metastate_->mutable_ctrl_pkt_bo(), ctrl_pkt_bo, cache_dir,
                    filename);
  }
  return *this;
}

MetaStateAPI &MetaStateAPI::update_dd_config(const DDConfig &cfg) {
  metastate_->clear_dd_config();
  metastate_->mutable_dd_config()->set_profile(cfg.profile);
  metastate_->mutable_dd_config()->set_pm_swap(cfg.pm_swap);
  metastate_->mutable_dd_config()->set_optimize_scratch(cfg.optimize_scratch);
  metastate_->mutable_dd_config()->set_eager_mode(cfg.eager_mode);
  metastate_->mutable_dd_config()->set_cache_dir(cfg.cache_dir);
  metastate_->mutable_dd_config()->set_model_name(cfg.model_name);
  return *this;
}

void MetaStateAPI::save_txt(const std::string &filename) const {
  std::string str;
  gpb::TextFormat::PrintToString(*metastate_, &str);
  std::ofstream ofs(filename);
  ofs << str;
}

void MetaStateAPI::save_json(const std::string &filename) const {
  std::string str;
  gpb::util::JsonPrintOptions options;
  options.add_whitespace = true;
  options.always_print_primitive_fields = true;
  auto status = gpb::util::MessageToJsonString(*metastate_, &str, options);
  std::ofstream ofs(filename);
  ofs << str;
}

void MetaStateAPI::save_bin(const std::string &filename) const {
  std::string s;
  metastate_->SerializeToString(&s);
  if (!save_func_) { // save to disk
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(s.data(), s.size());
  } else { // save to memory
    auto tmpfile = Utils::create_tmpfile();
    Utils::dump_to_tmpfile(tmpfile, s.data(), s.size());
    save_func_(filename, tmpfile);
  }
}

MetaStateAPI::MetaStateAPI(const std::string &metastate_file,
                           load_function load_func) {
  load_func_ = load_func;
  metastate_ = std::make_shared<dd_proto::MetaState>();
  if (!load_func_) {
    std::ifstream ifs(metastate_file, std::ios::binary);
    DD_ASSERT(ifs.good(), "Failed to open metstate file: "s + metastate_file);

    DD_ASSERT(metastate_->ParseFromIstream(&ifs),
              "Error parsing metastate file: "s + metastate_file);
  } else {
    // tried parse from file descriptor, didn't work
    auto file = load_func_(metastate_file);
    auto data = Utils::binary_io<char>::slurp_binary(file);
    DD_ASSERT(metastate_->ParseFromArray(data.data(), data.size()),
              "Error parsing metastate file: "s + metastate_file);
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Loaded metatestate successfully"));
}

Metadata MetaStateAPI::extract_meta() const {
  Metadata meta;
  meta.major_version = metastate_->major_version();
  meta.minor_version = metastate_->minor_version();

  // Op List
  meta.op_list.reserve(metastate_->op_list_size());
  for (const auto &proto_op : metastate_->op_list()) {
    Metadata::OpInfo op_info;
    op_info.name = proto_op.name();
    op_info.type = proto_op.type();
    op_info.pdi_id = proto_op.pdi_id();
    op_info.in_args.insert(op_info.in_args.end(), proto_op.in_args().begin(),
                           proto_op.in_args().end());
    op_info.const_args.insert(op_info.const_args.end(),
                              proto_op.const_args().begin(),
                              proto_op.const_args().end());
    op_info.out_args.insert(op_info.out_args.end(), proto_op.out_args().begin(),
                            proto_op.out_args().end());
    op_info.txn_bin.insert(op_info.txn_bin.end(),
                           proto_op.txn_bin().data().begin(),
                           proto_op.txn_bin().data().end());

    // ctrl_pkt_patch_info support only from meta version 1.1
    if ((meta.major_version >= 1) && (meta.minor_version >= 1)) {
      op_info.ctrl_pkt_patch_info.reserve(proto_op.ctrl_pkt_patch_info_size());
      for (const auto &proto_ctrl_pkt : proto_op.ctrl_pkt_patch_info()) {
        CtrlPktPatchInfo tctrl_pkt;
        tctrl_pkt.size = proto_ctrl_pkt.size();
        tctrl_pkt.offset = proto_ctrl_pkt.offset();
        tctrl_pkt.xrt_arg_idx = proto_ctrl_pkt.xrt_arg_idx();
        tctrl_pkt.bo_offset = proto_ctrl_pkt.bo_offset();
        op_info.ctrl_pkt_patch_info.push_back(std::move(tctrl_pkt));
      }
    }

    // Read Attrs TODO
    for (const auto &[name, proto_attr] : proto_op.attrs()) {
      if (proto_attr.type_url() == "type.googleapis.com/dd_proto.IntVector") {
        op_info.attr[name] =
            pbany_to_stdanyvec<int, dd_proto::IntVector>(proto_attr);
      } else if (proto_attr.type_url() ==
                 "type.googleapis.com/dd_proto.FloatVector") {
        op_info.attr[name] =
            pbany_to_stdanyvec<float, dd_proto::FloatVector>(proto_attr);
      } else if (proto_attr.type_url() ==
                 "type.googleapis.com/dd_proto.StringVector") {
        op_info.attr[name] =
            pbany_to_stdanyvec<std::string, dd_proto::StringVector>(proto_attr);
      } else if (proto_attr.type_url() ==
                 "type.googleapis.com/dd_proto.UInt32") {
        op_info.attr[name] =
            pbany_to_stdany<uint32_t, dd_proto::UInt32>(proto_attr);
      } else if (proto_attr.type_url() ==
                 "type.googleapis.com/dd_proto.String") {
        op_info.attr[name] =
            pbany_to_stdany<std::string, dd_proto::String>(proto_attr);
      } else {
        DD_THROW("Unsupported data type in attrs : " + proto_attr.type_url());
      }
    }

    meta.op_list.push_back(std::move(op_info));
  }

  // Fused Tensors
  for (const auto &[name, proto_tinfo] : metastate_->fused_tensors()) {
    Metadata::TensorInfo tinfo;
    tinfo.size = proto_tinfo.size();
    tinfo.arg_idx = proto_tinfo.xrt_arg_idx();
    tinfo.packed_tensors.insert(tinfo.packed_tensors.end(),
                                proto_tinfo.packed_tensors().begin(),
                                proto_tinfo.packed_tensors().end());
    meta.fused_tensors[name] = std::move(tinfo);
  }

  // Tensor Map
  for (const auto &[name, proto_tinfo] : metastate_->tensor_map()) {
    Metadata::OffsetInfo tinfo;
    tinfo.parent_name = proto_tinfo.parent_name();
    tinfo.offset = proto_tinfo.offset();
    tinfo.arg_idx = proto_tinfo.xrt_arg_idx();
    tinfo.dtype = proto_tinfo.dtype();
    tinfo.size_in_bytes = proto_tinfo.size_in_bytes();
    tinfo.shape.insert(tinfo.shape.end(), proto_tinfo.shape().begin(),
                       proto_tinfo.shape().end());
    if (proto_tinfo.has_file_name()) {
      tinfo.file_name = proto_tinfo.file_name();
      tinfo.file_size = proto_tinfo.file_size();
    }
    meta.tensor_map[name] = std::move(tinfo);
  }

  // Super Instr Map
  for (const auto &[name, proto_span] : metastate_->super_instr_map()) {
    Metadata::Span tspan;
    tspan.size = proto_span.size();
    tspan.offset = proto_span.offset();
    meta.super_instr_map[name] = std::move(tspan);
  }

  // Const Map
  for (const auto &[name, proto_span] : metastate_->const_map()) {
    Metadata::Span tspan;
    tspan.size = proto_span.size();
    tspan.offset = proto_span.offset();
    meta.const_map[name] = std::move(tspan);
  }

  // ctrl_pkt map is supported starting meta version 1.1
  if ((meta.major_version >= 1) && (meta.minor_version >= 1)) {
    for (const auto &[name, proto_span] : metastate_->control_pkt_map()) {
      Metadata::Span tspan;
      tspan.size = proto_span.size();
      tspan.offset = proto_span.offset();
      meta.ctrl_pkt_map[name] = std::move(tspan);
    }
  }

  meta.scratch_op_set.insert(metastate_->scratch_op_set().begin(),
                             metastate_->scratch_op_set().end());
  meta.max_op_scratch_pad_size = metastate_->max_op_scratch_pad_size();
  meta.max_tensor_padding_sz = metastate_->max_tensor_padding_sz();
  meta.json_path = metastate_->json_path();

  // Partition
  meta.partitions.reserve(metastate_->partitions_size());
  for (const auto &proto_part : metastate_->partitions()) {
    Partition tpartition;
    tpartition.op_range =
        std::make_pair(proto_part.start_idx(), proto_part.end_idx());
    tpartition.pdi_id = proto_part.pdi_id();
    meta.partitions.push_back(std::move(tpartition));
  }

  return meta;
}

FILE *MetaStateAPI::extract_const_bo(const std::string &cache_dir) const {
  return get_buffer_file(metastate_->const_bo(), cache_dir);
}

FILE *MetaStateAPI::extract_superinstr_bo(const std::string &cache_dir) const {
  return get_buffer_file(metastate_->superinstr_bo(), cache_dir);
}

FILE *MetaStateAPI::extract_input_bo(const std::string &cache_dir) const {
  return get_buffer_file(metastate_->input_bo(), cache_dir);
}

FILE *MetaStateAPI::extract_ctrl_pkt_bo(const std::string &cache_dir) const {
  if (!(metastate_->major_version() >= 1) ||
      !(metastate_->minor_version() >= 1)) {
    return {};
  }
  auto data = get_buffer_file(metastate_->ctrl_pkt_bo(), cache_dir);
  return data;
}

DDConfig MetaStateAPI::extract_dd_config() const {
  DDConfig cfg;
  cfg.profile = metastate_->dd_config().profile();
  cfg.pm_swap = metastate_->dd_config().pm_swap();
  cfg.optimize_scratch = metastate_->dd_config().optimize_scratch();
  cfg.eager_mode = metastate_->dd_config().eager_mode();
  cfg.model_name = metastate_->dd_config().model_name();
  return cfg;
}

} // namespace OpsFusion
