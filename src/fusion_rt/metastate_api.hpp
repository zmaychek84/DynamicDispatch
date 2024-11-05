#ifndef __METASTATEAPI_HPP__
#define __METASTATEAPI_HPP__

#include <fusion_rt/metastate.pb.h>
#include <op_fuser/fuse_types.hpp>
#include <op_fuser/fusion_rt.hpp>

namespace OpsFusion {

struct MetaStateAPI {
  // Meta To MetaState
  MetaStateAPI();
  void set_buffer_file(dd_proto::BinFile *binfile, FILE *file,
                       const std::string &dir,
                       const std::string &filename) const;
  FILE *get_buffer_file(const dd_proto::BinFile &binfile,
                        const std::string &dir) const;
  void save_txt(const std::string &filename) const;
  void save_json(const std::string &filename) const;
  void save_bin(const std::string &filename) const;

  MetaStateAPI &update_meta(const Metadata &meta);
  MetaStateAPI &update_save_func(save_function func);
  MetaStateAPI &update_const_bo(FILE *const_bo, const std::string &cache_dir,
                                const std::string &filename);
  MetaStateAPI &update_superinstr_bo(FILE *superinstr_bo,
                                     const std::string &cache_dir,
                                     const std::string &filename);
  MetaStateAPI &update_input_bo(FILE *input_bo, const std::string &cache_dir,
                                const std::string &filename);
  MetaStateAPI &update_ctrl_pkt_bo(FILE *ctrl_pkt_bo,
                                   const std::string &cache_dir,
                                   const std::string &filename);
  MetaStateAPI &update_dd_config(const DDConfig &cfg);
  static MetaStateAPI create();

  // MetaState To Meta
  MetaStateAPI(const std::string &metastate_file,
               load_function load_func = nullptr);
  Metadata extract_meta() const;
  FILE *extract_const_bo(const std::string &cache_dir) const;
  FILE *extract_superinstr_bo(const std::string &cache_dir) const;
  FILE *extract_input_bo(const std::string &cache_dir) const;
  FILE *extract_ctrl_pkt_bo(const std::string &cache_dir) const;
  DDConfig extract_dd_config() const;

private:
  std::shared_ptr<dd_proto::MetaState> metastate_;
  save_function save_func_;
  load_function load_func_;
};

} // namespace OpsFusion

#endif
