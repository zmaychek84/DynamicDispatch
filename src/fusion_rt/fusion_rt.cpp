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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <mutex>

#include <op_fuser/fuse_ops.hpp>
#include <op_fuser/fusion_rt.hpp>
#include <ops/op_builder.hpp>
#include <ops/record_timer/record_timer.hpp>
#include <ops/xcom/subgraph/subgraph.hpp>
// #include "aiebu_assembler.h"

#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/platform.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

#include "metastate_api.hpp"
#include "passes/passes.hpp"
#include "txn/txn_utils.hpp"
#include "utils/ctrl_pkt_utils.hpp"
#include "utils/dpu_mdata.hpp"
#include <utils/tmpfile.hpp>

#include "md5.h"
#include "weak.hpp"
#include <experimental/xrt_error.h>
namespace fs = std::filesystem;

#ifdef SIMNOWLITE_EN
static constexpr uint32_t HOST_BO_GROUP_ID = 8;
#else
static constexpr uint32_t HOST_BO_GROUP_ID = 0;
#endif

static constexpr size_t XRT_BO_MIN_SIZE = 4096; // Bytes
static constexpr size_t XRT_BO_INIT_VALUE = 0;
// 64 MB support per hw context
static constexpr size_t INSTR_XRT_BO_MAX_SIZE = 64ULL * 1024ULL * 1024ULL;
// space used up by all PDI's in loaded xclbin, and XRT/XDP
static constexpr size_t INSTR_XRT_BO_RESERVE_SIZE = 4ULL * 1024ULL * 1024ULL;
// ping-pong to hide latency of copying over instruction during execution
static constexpr size_t NUM_STATIC_INSTR_BUFFERS = 2;
// need to account for alignment to estimate size (ignores fragmentation)
static constexpr size_t INSTR_XRT_BO_ALIGNMENT = 32ULL * 1024ULL;

// fallback buffers - shared by all FusionRuntime objects on same context
static const size_t INSTR_XRT_BO_STACK_SIZE =
    std::stoull(Utils::get_env_var("DD_ENV_INSTR_STACK_SIZE_MB",
                                   std::to_string(8ULL))) *
    1024ULL * 1024ULL;
static const size_t INSTR_BUFFER_SIZE =
    INSTR_XRT_BO_STACK_SIZE / NUM_STATIC_INSTR_BUFFERS;
// how much we can dynamically allocate
static const size_t INSTR_XRT_BO_HEAP_SIZE =
    INSTR_XRT_BO_MAX_SIZE - INSTR_XRT_BO_RESERVE_SIZE - INSTR_XRT_BO_STACK_SIZE;

struct XRTBufferState {
  // how much memory is currently used by this context
  size_t heap_total_size = 0;
  // for debug - how many BOs have already been allocated
  size_t num_instr_bos = 0;
  // global instr BO per xrt hw context
  std::vector<xrt::bo> static_instr_bos;
  // actual size of instruction contents
  std::vector<size_t> static_instr_sizes;
  // How many FusionRT instances associated with this context
  size_t num_fusionrt_instances = 0;
};

// NOTE: current access to this is NOT THREAD SAFE!!!
static std::map<xrt_core::hwctx_handle *, XRTBufferState> xrt_instr_state;
static std::mutex instr_state_mutex;

static bool enable_write_internal_bufs = static_cast<bool>(
    std::stol(Utils::get_env_var("DD_WRITE_INTERNAL_BUFS", "0")));

static size_t compute_hash(const void *data, size_t size) {
  std::hash<std::string_view> hash;
  return hash(std::string_view((const char *)data, size));
}

static std::string replace_characters(const std::string &name,
                                      const std::string &pattern,
                                      char replacement) {
  std::string::size_type pos{0};
  std::string res(name);
  while (true) {
    pos = name.find_first_of(pattern, pos);
    if (pos == std::string::npos) {
      break;
    }
    res[pos] = replacement;
    pos++;
  }
  return res;
}

static std::vector<std::string> filter_xrt_kernels(const xrt::xclbin &xclbin) {

  std::vector<std::string> kernel_names;
  for (const auto &kernel : xclbin.get_kernels()) {
    const auto kernel_name = kernel.get_name();

    // only avoid adding profiling kernel, vadd and xcompiler DPUs
    if ((kernel_name.rfind("XDP_KERNEL", 0) != 0) &&
        (kernel_name.rfind("vadd", 0) != 0) &&
        (kernel_name.rfind("DPU_PDI", 0) != 0)) {
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "FusionRuntime : found kernel : {}", kernel_name));
      size_t elf_idx = kernel_name.find("_ELF");
      if (elf_idx != -1) {
        auto non_elf_kernel = kernel_name.substr(0, elf_idx);
        auto it =
            std::find(kernel_names.begin(), kernel_names.end(), non_elf_kernel);
        if (it != kernel_names.end()) {
          kernel_names[it - kernel_names.begin()] = kernel_name;
          continue;
        }
      } else {
        auto it = std::find(kernel_names.begin(), kernel_names.end(),
                            kernel_name + "_ELF");
        if (it != kernel_names.end()) {
          continue;
        }
      }

      kernel_names.push_back(kernel_name);
    }
  }
  // In case kernel names arent in order within xclbin
  std::sort(kernel_names.begin(), kernel_names.end());

  return kernel_names;
}

static void create_kernel_name_to_pdi_map(
    const std::vector<std::string> &kernel_names,
    std::map<std::string, std::uint32_t> &kernel_name_to_pdi_idx) {
  std::uint32_t start_pdi_idx = 0;
  for (const auto &kernel_name : kernel_names) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Creating kernel to pdi_id mappding : {} {}",
        kernel_name, start_pdi_idx));
    kernel_name_to_pdi_idx[kernel_name] = start_pdi_idx++;
  }
}

namespace OpsFusion {

// Depad & Copy data from src to dst buffer.
// So src is larger than
static void copy_data(const Tensor &src_tensor, const Tensor &dst_tensor) {
  size_t src_sz =
      std::accumulate(src_tensor.shape.begin(), src_tensor.shape.end(),
                      size_t{1}, std::multiplies{}) *
      Utils::get_size_of_type(src_tensor.dtype);
  memcpy(dst_tensor.data, src_tensor.data, src_sz);
}

static void write_to_bo(xrt::bo &dst_bo, size_t offset, const void *src,
                        size_t size) {
  dst_bo.write(src, size, offset);
  dst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

static void read_from_bo(xrt::bo &src_bo, size_t offset, void *dst,
                         size_t size) {
  src_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  src_bo.read(dst, size, offset);
}

template <typename T> static float sum_bo(xrt::bo &src) {
  T *ptr = src.map<T *>();
  size_t size = src.size() / sizeof(T);
  return std::accumulate(ptr, ptr + size, 0.0f,
                         [](float a, T b) { return a + b; });
}

template <typename T> static int sum_bo_int(xrt::bo &src) {
  T *ptr = src.map<T *>();
  size_t size = src.size() / sizeof(T);
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += (int)ptr[i];
  }
  return sum;
}

FusionRuntime::FusionRuntime() : cpu_only_runtime_(true) {}

FusionRuntime::FusionRuntime(const std::string &xclbin_filename)
    : FusionRuntime(xclbin_filename,
                    OpsFusion::read_bin_file<char>(xclbin_filename)) {}
FusionRuntime::FusionRuntime(const std::string &xclbin_filename,
                             const std::vector<char> &xclbin_content,
                             const std::string &kernel_name_prefix,
                             const std::map<std::string, std::uint32_t> &qos)
    : use_external_ctx_(false), xclbin_filename_(xclbin_filename),
      xclbin_content_(std::move(xclbin_content)), qos_(qos) {}

FusionRuntime::FusionRuntime(xrt::hw_context *ctx,
                             const std::string &kernel_name_prefix)
    : use_external_ctx_(true), ctx_(*ctx) {}

xrt::bo FusionRuntime::allocate_xrt_buffer(const xrt::hw_context &ctx,
                                           const size_t &sz,
                                           xrt::bo::flags flag,
                                           xrt::memory_group grp) {
  if (elf_flow_) {
    return xrt::ext::bo(ctx, sz);
  }
  return xrt::bo(ctx, sz, flag, grp);
}

bool FusionRuntime::is_elf_kernel(const std::string &kernel_name) {
  return kernel_name.find("ELF") != -1;
}
void FusionRuntime::initialize_runtime() {

  if (!use_external_ctx_) {
    xrt_ctx_ = ryzenai::dynamic_dispatch::xrt_context::get_instance(
        xclbin_filename_, 0, qos_, xclbin_content_);
    ctx_ = xrt_ctx_->get_context();
  }

  const auto kernel_names = filter_xrt_kernels(ctx_.get_xclbin());
  elf_flow_ = std::any_of(kernel_names.begin(), kernel_names.end(),
                          [](const std::string &kernel_name) {
                            return FusionRuntime::is_elf_kernel(kernel_name);
                          });
  create_kernel_name_to_pdi_map(kernel_names, kernel_name_to_pdi_idx_);
}

std::string
FusionRuntime::get_canonicalize_kernel_name(const OpPDIMap &op_pdi_map,
                                            int pdi_index) {
  std::string kernel_name = op_pdi_map.pdi_id_to_kernel_map.at(pdi_index);
  std::string postfix = "";
  if (elf_flow_ && !FusionRuntime::is_elf_kernel(kernel_name)) {
    postfix = "_ELF";
  }
  return kernel_name + postfix;
}

void FusionRuntime::initialize_kernels(
    const Metadata &meta,
    const std::map<std::string, std::uint32_t> &kernel_name_to_pdi_idx) {
  auto part_size = meta.partitions.size();
  kernels_.resize(part_size);
  runs_.resize(part_size);
  // transactions bins are aligned with partitions and subgraph kernels.
  for (int i = 0; i < part_size; i++) {
    auto idx = meta.partitions[i].pdi_id;
    auto part_data =
        use_xclbin_parse_data_ ? op_pdi_map_ : DEFAULT_OP_TO_PDI_MAP_;
    std::string kernel_name = get_canonicalize_kernel_name(part_data, idx);
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Creating kernel object : {}", kernel_name));
    xrt::kernel k;
    if (!FusionRuntime::is_elf_kernel(kernel_name)) {
      k = xrt::kernel(ctx_, kernel_name);
    } else {
      if (i >= subgraph_elfs_mod_.size()) {
        DD_THROW("Kernels and ELFs modules are not aligned.");
      }
      elf_flow_ = true;
      k = xrt::ext::kernel(ctx_, subgraph_elfs_mod_[i], kernel_name);
    }

    // kernels and partitions are aligned with each other.
    kernels_[i] = k;
    runs_[i] = xrt::run(k);
  }

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  std::call_once(logger_flag_, []() {
    std::string header =
        "Graph/PDI_Partition, xrt execute time(ns), "
        "in_copy_time(ns), in_sync_time(ns), "
        "out_copy_time(ns), out_sync_time(ns), json_path, oplist_str\n";
    RYZENAI_LOG_INFO(header);
  });

  // make inserting into global state here thread safe
  std::lock_guard<std::mutex> guard(instr_state_mutex);

  if (xrt_instr_state.find(handle) == xrt_instr_state.end()) {
    xrt_instr_state[handle] = XRTBufferState{};
    for (size_t i = 0; i < NUM_STATIC_INSTR_BUFFERS; i++) {
      xrt_instr_state.at(handle).static_instr_bos.emplace_back(
          xrt::bo(ctx_, INSTR_BUFFER_SIZE, xrt::bo::flags::cacheable,
                  kernels_[0].group_id(1)));
      xrt_instr_state.at(handle).static_instr_sizes.push_back(0);
    }
    xrt_instr_state.at(handle).num_instr_bos = NUM_STATIC_INSTR_BUFFERS;
  }
  xrt_instr_state.at(handle).num_fusionrt_instances++;
}

FusionRuntime::~FusionRuntime() {
  if (cpu_only_runtime_) {
    return;
  }

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  // make book-keeping here thread safe
  // heap_total_size is used to determine if we should
  // use static instruction BO or not
  std::lock_guard<std::mutex> guard(instr_state_mutex);
  MAP_AT(xrt_instr_state, handle).num_fusionrt_instances--;

  for (auto &instr_bo : instr_bos_) {
    xrt_instr_state.at(handle).heap_total_size -=
        Utils::align_to_next(instr_bo.size(), INSTR_XRT_BO_ALIGNMENT);
  }
  xrt_instr_state.at(handle).num_instr_bos -= instr_bos_.size();

  if (xrt_instr_state.at(handle).num_fusionrt_instances == 0) {
    xrt_instr_state.erase(handle);
  }

  release_host_resources();
}

void FusionRuntime::execute(const std::vector<Tensor> &inputs,
                            const std::vector<Tensor> &outputs) {
  // this lock guard serves 2 purposes
  //  - Make copying in inputs and outputs out thread-safe
  //    when there is single FusionRuntime object, but multiple theads call
  //    execute
  //  - For multiple partitions, which would need to be
  //    run together
  std::lock_guard<std::mutex> guard(execute_mutex_);
  const auto &meta = meta_;
  if (scratch_bo_allocate_) {
    scratch_bo_ =
        allocate_xrt_buffer(ctx_, scratch_bo_sz_, xrt::bo::flags::host_only,
                            kernels_[0].group_id(HOST_BO_GROUP_ID));
    patch_ctrl_pkt(meta);
    if (!elf_flow_) {
      setup_xrt_run(meta);
    }
    scratch_bo_allocate_ = false;
  }
  merge_inputs(inputs, meta);

  if (enable_write_internal_bufs) {
    unpack_internal_buffers("tmp/dd_bufs_pre_exec");
  }
  xrt_exec_time_ = 0;
  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);
  size_t cache_instr_idx = 0;

  const OpPDIMap &op_pdi_map =
      use_xclbin_parse_data_ ? op_pdi_map_ : DEFAULT_OP_TO_PDI_MAP_;

  int pdi_id = 0;
  int pdi_idx = 0;

  if (use_instr_sw_cache_) {
    // NOTE: this writes to BO and does sync
    if (!elf_flow_) {
      std::lock_guard<std::mutex> guard(instr_state_mutex);
      write_to_bo(xrt_instr_state.at(handle).static_instr_bos[cache_instr_idx],
                  0, /*offset*/
                  fused_instr_vec_.at(0).data(), fused_instr_vec_.at(0).size());
      xrt_instr_state.at(handle).static_instr_sizes[cache_instr_idx] =
          fused_instr_vec_.at(0).size();
    }

    for (size_t partition_idx = 0; partition_idx < meta.partitions.size();
         partition_idx++) {
      auto exec_start = GET_ELAPSED_TIME_NS();
      size_t instr_idx = cache_instr_idx;
      cache_instr_idx = (cache_instr_idx + 1) % NUM_STATIC_INSTR_BUFFERS;
      bool prefetch_instr = (partition_idx != (meta.partitions.size() - 1));
      auto pdi_id = meta.partitions[partition_idx].pdi_id;

      try {
        if (!elf_flow_) {
          runs_[partition_idx].set_arg(
              1, xrt_instr_state.at(handle).static_instr_bos[instr_idx]);
          runs_[partition_idx].set_arg(
              2, xrt_instr_state.at(handle).static_instr_sizes[instr_idx] /
                     sizeof(int));
        }
        runs_[partition_idx].start();
        // try to overlap instruction copying with AIE execution
        if (prefetch_instr && !elf_flow_) {
          write_to_bo(
              xrt_instr_state.at(handle).static_instr_bos[cache_instr_idx],
              0, /*offset*/
              fused_instr_vec_.at(partition_idx + 1).data(),
              fused_instr_vec_.at(partition_idx + 1).size());
          xrt_instr_state.at(handle).static_instr_sizes[cache_instr_idx] =
              fused_instr_vec_.at(partition_idx + 1).size();
        }
        runs_[partition_idx].wait2();
      } catch (const std::exception &e) {
        if (enable_write_internal_bufs) {
          unpack_internal_buffers("tmp/dd_buf_post_hang");
        }
#ifdef RYZENAI_DEBUG
        std::cout << "Running under debug mode...  Hardware context handle = "
                  << ctx_.get_handle() << ", PID = " << Platform::get_pid()
                  << std::endl;
        std::cout << "Will wait for user input." << std::endl;
        std::cin.get();
#endif

        std::cerr << "ERROR: Kernel partition: " << partition_idx
                  << ", pdi_id: " << (std::uint32_t)pdi_id << " timed out!"
                  << std::endl;
        std::cerr << "Details: " << e.what() << std::endl;
        if (cfg_.eager_mode) {
          std::cout << "  Op ID : " << partition_idx << "\n"
                    << "  Op Name : " << meta.op_list.at(partition_idx).name
                    << "\n"
                    << "  Op Type : " << meta.op_list.at(partition_idx).type
                    << std::endl;
        } else {
          std::cout << "  Enable eager-mode for more verbosity." << std::endl;
        }

        xrt::error err = xrt::error(ctx_.get_device(), XRT_ERROR_CLASS_AIE);
        if (err.get_error_code()) {
          std::string err_message =
              std::string("Error while executing pdi_id: ") +
              std::to_string((std::uint32_t)pdi_id) +
              ", partition: " + std::to_string(partition_idx) +
              ", info: " + err.to_string();
          std::cerr << err_message << std::endl;
          RYZENAI_LOG_TRACE(err_message);
        }

        DD_THROW(OpsFusion::dd_format(
            "Kernel partition: {} pdi_id: {} timeout (Detail : {})",
            partition_idx, (std::uint32_t)pdi_id, e.what()));
      }
      auto exec_end = GET_ELAPSED_TIME_NS();
      int64_t partition_exec_time = static_cast<int64_t>(exec_end - exec_start);
      xrt_exec_time_ += partition_exec_time;

#ifdef RYZENAI_PERF
      std::string oplist_str = "";
      for (auto ind = meta.partitions[partition_idx].op_range.first;
           ind < meta.partitions[partition_idx].op_range.second; ind++) {
        oplist_str += meta.op_list[ind].name + std::string(",");
      }
#endif
      RYZENAI_LOG_INFO("PDI_Partition " + std::to_string(partition_idx) +
                       " , " + std::to_string(partition_exec_time) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       meta.json_path + ", " + oplist_str + "\n");
    }
  } else {
    auto &meta = get_meta();
    for (size_t partition_idx = 0; partition_idx < meta.partitions.size();
         partition_idx++) {
      auto exec_start = GET_ELAPSED_TIME_NS();
      size_t instr_idx = partition_idx;
      auto pdi_id = meta.partitions[partition_idx].pdi_id;

      try {
        if (!elf_flow_) {
          runs_[partition_idx].set_arg(1, instr_bos_[instr_idx]);
          runs_[partition_idx].set_arg(2, instr_bos_[instr_idx].size() /
                                              sizeof(int));
        }
        runs_[partition_idx].start();
        runs_[partition_idx].wait2();
      } catch (const std::exception &e) {
        if (enable_write_internal_bufs) {
          unpack_internal_buffers("tmp/dd_buf_post_hang");
        }
#ifdef RYZENAI_DEBUG
        std::cout << "Running under debug mode...  Hardware context handle = "
                  << ctx_.get_handle() << ", PID = " << Platform::get_pid()
                  << std::endl;
        std::cout << "Will wait for user input." << std::endl;
        std::cin.get();
#endif

        std::cerr << "ERROR: Kernel partition: " << partition_idx
                  << ", pdi_id: " << (std::uint32_t)pdi_id << " timed out!"
                  << std::endl;
        std::cerr << "Details: " << e.what() << std::endl;
        if (cfg_.eager_mode) {
          std::cout << "  Op ID : " << partition_idx << "\n"
                    << "  Op Name : " << meta.op_list.at(partition_idx).name
                    << "\n"
                    << "  Op Type : " << meta.op_list.at(partition_idx).type
                    << std::endl;
        } else {
          std::cout << "  Enable eager-mode for more verbosity." << std::endl;
        }

        xrt::error err = xrt::error(ctx_.get_device(), XRT_ERROR_CLASS_AIE);
        if (err.get_error_code()) {
          std::string err_message =
              std::string("Error while executing pdi_id: ") +
              std::to_string((std::uint32_t)pdi_id) +
              ", partition: " + std::to_string(partition_idx) +
              ", info: " + err.to_string();
          std::cerr << err_message << std::endl;
          RYZENAI_LOG_TRACE(err_message);
        }

        DD_THROW(OpsFusion::dd_format(
            "Kernel partition: {} pdi_id: {} timeout (Detail : {})",
            partition_idx, (std::uint32_t)pdi_id, e.what()));
      }
      auto exec_end = GET_ELAPSED_TIME_NS();
      int64_t partition_exec_time = static_cast<int64_t>(exec_end - exec_start);
      xrt_exec_time_ += partition_exec_time;

#ifdef RYZENAI_PERF
      std::string oplist_str = "";
      for (auto ind = meta.partitions[partition_idx].op_range.first;
           ind < meta.partitions[partition_idx].op_range.second; ind++) {
        oplist_str += meta.op_list[ind].name + std::string(",");
      }
#endif
      RYZENAI_LOG_INFO("PDI_Partition " + std::to_string(partition_idx) +
                       " , " + std::to_string(partition_exec_time) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       std::to_string(0) + ", " + std::to_string(0) + ", " +
                       meta.json_path + ", " + oplist_str + "\n");
    }
  }

  if (enable_write_internal_bufs) {
    unpack_internal_buffers("tmp/dd_bufs_post_exec");
  }
  split_outputs(outputs, meta);

  RYZENAI_LOG_INFO("Graph, " + std::to_string(xrt_exec_time_) + ", " +
                   std::to_string(input_copy_time_) + ", " +
                   std::to_string(input_sync_time_) + ", " +
                   std::to_string(output_copy_time_) + ", " +
                   std::to_string(output_sync_time_) + ", " + meta.json_path +
                   "\n");
}

/**
 * @brief utility function to write the contents of an xrt BO to file
 *
 * @param bo  xrt buffer object
 * @param fname filename to write the contents of the BO.
 */
void FusionRuntime::save_bo(xrt::bo &bo, const std::string filename) {
  auto bo_map = bo.map<uint8_t *>();
  std::ofstream ofs(filename, std::ios::binary);
  DD_ASSERT(ofs, OpsFusion::dd_format("Couldn't open {} for writing", filename))
  ofs.write((char *)bo_map, bo.size());
  ofs.close();
}

/**
 * @brief save contents of buffer objects in cache directory.
 *
 */
void FusionRuntime::save_buffer_objects() {
  std::string input_bo_fname = cfg_.cache_dir + "./input_bo_fname.bin";
  std::string const_bo_fname = cfg_.cache_dir + "./const_bo_fname.bin";
  std::string super_instr_bo_fname =
      cfg_.cache_dir + "./super_instr_bo_fname.bin";

  /**
   * all bos are being saved as files in the cache directory.
   * at runtime, xrt bos will be populated with the contents of these files
   * input_bo_ are saved as input buffers may need some initialiation based on
   * the operator. const_bo_ contains the weights / constant inputs for all
   * operators. super_instr_bo_ contains all the super kernel params for
   * operators.
   */
  save_bo(input_bo_, input_bo_fname);
  save_bo(*const_bo_, const_bo_fname);
  save_bo(super_instr_bo_, super_instr_bo_fname);
}

void FusionRuntime::compile(const Metadata &meta, const std::string &base_dir,
                            const DDConfig &cfg,
                            std::map<std::string, SimpleSpan> const_map) {
  // TODO : Need a way to compare if metadata is same as old, and if so skip.

  RYZENAI_LOG_TRACE("FusionRuntime : Init ...");
  meta_ = meta;
  cfg_ = cfg;

  // Update .const filenames in meta with absolute paths.
  // This is done to enable relocatable cache.
  // Meta contains only filenames and full path is assembled here.
  // Note : Cache relocation is applicable to VAIP C++ EP Flow only.
  // All other flow, it expects absolute paths in the meta.json
  if (!cfg_.cache_dir.empty()) {
    fs::path cache_dir_path = fs::path(cfg_.cache_dir);
    for (auto &[tname, tinfo] : meta_.tensor_map) {
      if (tinfo.parent_name == "const") {
        if (tinfo.file_name.empty() ||
            fs::path(tinfo.file_name).is_absolute()) {
          continue;
        }
        auto new_path = cache_dir_path / tinfo.file_name;
        tinfo.file_name = new_path.string();
      }
    }
  }

  // Map of tensorname -> buffer
  //  auto const_map = const_db.empty() ? MetaUtils::load_const_buffers(meta_)
  //                                    : std::move(const_db);
  std::map<std::string, std::vector<char>> file_const_map;
  if (const_map.empty()) {
    RYZENAI_LOG_TRACE("const_db is empty, loading from file");
    file_const_map = MetaUtils::load_const_buffers(meta_);
    for (const auto &item : file_const_map) {
      const_map[item.first] = {const_cast<char *>(item.second.data()),
                               item.second.size()};
    }
  } else {
    RYZENAI_LOG_TRACE("skipping loading from file");
  }
  // if env variables are set, update cfg_
  // check if profile option is enabled using env varaibles
  auto profile_level = Utils::get_env_var("DD_ENABLE_PROFILE");
  if (profile_level != "") {
    auto p_lvl = std::stoi(profile_level);
    cfg_.profile = std::min(3, p_lvl);
  }

  RYZENAI_LOG_TRACE(dd_format("Setting profile level to {}", cfg_.profile));

  OpInterface::set_dd_base_dir(base_dir);

  const std::string &model_name = cfg_.model_name;

  use_xclbin_parse_data_ =
      parse_xclbin_metadata(op_pdi_map_, model_name, cfg_.xclbin_content);

  // partition ops to PDIs
  assign_pdi_id_pass(
      use_xclbin_parse_data_ ? op_pdi_map_ : DEFAULT_OP_TO_PDI_MAP_, meta_);

  if (cfg_.pm_swap) {
    meta_ = insert_pm_swap_nodes(meta_);
  }

  generate_pdi_partitions_pass(meta_, cfg_.eager_mode);

  elf_flow_ = std::any_of(
      kernel_name_to_pdi_idx_.begin(), kernel_name_to_pdi_idx_.end(),
      [](auto it) { return FusionRuntime::is_elf_kernel(it.first); });
  meta_.aux_info["elf_flow"] = elf_flow_;

  // TODO : This is temporary flag, By default we will not insert any preemption
  // point Once preemption related issues are resolved , we will enable
  // inserting preemption by default'
  bool enable_preemption = Utils::get_env_var("ENABLE_PREEMPTION") == "1";

  if (elf_flow_) {
    if (enable_preemption) {
      size_t op_count = meta_.op_list.size();
      meta_ = insert_preemption_nodes(meta_);
      generate_pdi_partitions_pass(meta_, cfg_.eager_mode);
    } else {
      std::cout
          << "[WARNING] Inserting fine grained preemption is disabled. To "
             "insert fine grained preemption please set environment variable"
             " ENABLE_PREEMPTION to '1'"
          << std::endl;
    }
  }

  if (cfg_.profile) {
    meta_ = insert_record_timer_nodes(meta_, cfg_.profile);
    generate_pdi_partitions_pass(meta_, cfg_.eager_mode);
  }

  analyze_buffer_reqs(meta_);

  if (cfg_.optimize_scratch) {
    optimize_scratch_buffer(meta_);
  }

  allocate_host_bos(meta_);
  initialize_inputs(meta_);
  load_const(meta_, const_map);
  fill_super_instr(meta_, const_map);
  fill_ctrl_pkts(meta_);
  prepare_formatting_ops();

  relocate_ctrl_pkt_patch_info(meta_, elf_flow_);
  fetch_op_txn_bins(meta_, const_map, elf_flow_);
}

/**
 * @brief utility function to release all host file handlers after use.
 */
void FusionRuntime::release_host_resources() {
  if (input_vec_file_ptr_) {
    fclose(input_vec_file_ptr_);
  }

  if (const_vec_file_ptr_) {
    fclose(const_vec_file_ptr_);
  }

  if (super_instr_vec_file_ptr_) {
    fclose(super_instr_vec_file_ptr_);
  }

  if (ctrl_pkt_vec_file_ptr_) {
    fclose(ctrl_pkt_vec_file_ptr_);
  }
}

/**
 * @brief Saves fusion state.
 *
 * This function saves the state in cache_dir.
 *
 * @param state_name name of the state file.
 * @param save_func save function.
 * @return does not return anything.
 */
void FusionRuntime::save_state(const std::string &state_name,
                               save_function save_func) {

  std::lock_guard<std::mutex> guard(load_save_state_mutex_);
  auto cache_dir = cfg_.cache_dir;
  fs::path state_file = fs::path(cache_dir) / state_name;
  auto subg_name = state_file.filename();
  auto metastate =
      MetaStateAPI()
          .update_save_func(save_func)
          .update_meta(meta_)
          .update_const_bo(const_vec_file_ptr_, cache_dir,
                           subg_name.replace_extension("fconst").string())
          .update_superinstr_bo(super_instr_vec_file_ptr_, cache_dir,
                                subg_name.replace_extension("super").string())
          .update_input_bo(input_vec_file_ptr_, cache_dir,
                           subg_name.replace_extension("input").string())
          .update_ctrl_pkt_bo(ctrl_pkt_vec_file_ptr_, cache_dir,
                              subg_name.replace_extension("ctrlpkt").string())
          .update_dd_config(cfg_);

  metastate.save_bin(state_file.string());
  release_host_resources();
}

static std::unique_ptr<std::unique_lock<std::mutex>>
acquire_lock_for_const_bo() {
  static std::mutex mtx;
  return std::make_unique<std::unique_lock<std::mutex>>(mtx);
}
static std::string calculate_md5sum(FILE *fp) {
  auto original_pos = ftell64(fp);
  auto error_code = fseek64(fp, 0, SEEK_SET);
  DD_ASSERT(error_code == 0,
            OpsFusion::dd_format("fseek64 error: {}", error_code));
  auto buffer = std::vector<char>(1024 * 4);
  MD5 md5 = MD5();
  for (auto read_size = fread(buffer.data(), 1, buffer.size(), fp);
       read_size != 0; read_size = fread(buffer.data(), 1, buffer.size(), fp)) {
    md5.add(buffer.data(), read_size);
    // std::cout << "abc: " << read_size << std::endl;
  }
  error_code = fseek64(fp, original_pos, SEEK_SET);
  return md5.getHash();
}
/**
 * @brief Loads fusion state.
 *
 * This function loads fusion state.
 *
 * @param state_path full path to the state file.
 * @param load_func load function.
 * @return does not return anything.
 */
void FusionRuntime::load_state(const std::string &state_path,
                               load_function load_func) {
  std::lock_guard<std::mutex> guard(load_save_state_mutex_);
  fs::path state_file{state_path};
  auto cache_dir = state_file.parent_path().string();

  MetaStateAPI metastate(state_path, load_func);
  meta_ = metastate.extract_meta();
  const_vec_file_ptr_ = metastate.extract_const_bo(cache_dir);
  const_vec_file_md5_ = calculate_md5sum(const_vec_file_ptr_);
  super_instr_vec_file_ptr_ = metastate.extract_superinstr_bo(cache_dir);
  input_vec_file_ptr_ = metastate.extract_input_bo(cache_dir);
  ctrl_pkt_vec_file_ptr_ = metastate.extract_ctrl_pkt_bo(cache_dir);
  cfg_ = metastate.extract_dd_config();
}

const std::vector<xrt::module> &FusionRuntime::convert_to_elf(
    const std::vector<std::vector<uint8_t>> &fused_bins, const Metadata &meta,
    FILE *ctrl_pkt_vec_file_ptr) {

  if (!elf_flow_) {
    return subgraph_elfs_mod_;
  }
  subgraph_elfs_mod_.clear();
  auto asm_path =
      std::filesystem::path(Utils::get_env_var("AIEBU_ASM")).string();
  if (asm_path.size() == 0) {
    std::string error = "ERROR: Please set environment variable AIEBU_ASM, "
                        "Which should points to aiebu directory\n";
    throw std::runtime_error(error);
  }

  auto asm_exe =
      std::filesystem::path(asm_path + "\\bin\\aiebu-asm.exe").string();
  auto asm_lib = std::filesystem::path(asm_path + "\\lib\\aie2").string();

  std::string tempDir =
      std::filesystem::path(fs::temp_directory_path().string() + "tmpdir\\")
          .string();
  if (!fs::exists(tempDir)) {
    fs::create_directory(tempDir);
  }
  auto ctrl_pkt_json = meta_to_ctrl_pkt_json(meta);
  std::string ctr_pkt_cmd = "";
  if (ctrl_pkt_json.size() != 0) {
    auto time_stamp = Utils::generateCurrTimeStamp();
    std::string ctrl_pkt_bin_path =
        tempDir + "fused_ctrpkt_bin" + time_stamp + ".ctrlpkt";
    std::string ctr_pkt_json_path =
        tempDir + "fused_ctrpkt" + time_stamp + ".json";
    Utils::save_tmpfile_on_disk(ctrl_pkt_bin_path, ctrl_pkt_vec_file_ptr);
    std::ofstream jsonf(ctr_pkt_json_path);
    jsonf << std::setw(4) << ctrl_pkt_json << std::endl;
    jsonf.flush();
    jsonf.close();
    ctr_pkt_cmd = " -p " + ctrl_pkt_bin_path + " -j " + ctr_pkt_json_path;
  }
#ifdef _WIN32
  for (const auto &txn : fused_bins) {
    try {
      auto time_stamp = Utils::generateCurrTimeStamp();
      // ToDO move to DD's tmpFile utility.
      std::string txn_bin_path =
          tempDir + "input_" + std::to_string(txn.size()) + time_stamp + ".bin";
      std::string elf_path = tempDir + "output_" + std::to_string(txn.size()) +
                             time_stamp + ".elf";
      std::ofstream outFile(txn_bin_path, std::ios::out | std::ios::binary);
      outFile.write(reinterpret_cast<const char *>(txn.data()), txn.size());
      outFile.flush();
      outFile.close();
      std::string command = asm_exe + " -t aie2txn -c " +
                            std::filesystem::path(txn_bin_path).string() +
                            ctr_pkt_cmd + " -o " +
                            std::filesystem::path(elf_path).string() +
                            " -l preempt -L " + asm_lib;

      RYZENAI_LOG_TRACE("Executing ... " + command + "\n");
      {

        STARTUPINFOA si = {sizeof(STARTUPINFOA)}; // Initialize STARTUPINFO
        PROCESS_INFORMATION pi = {};
        BOOL success = CreateProcessA(
            NULL, // Application name (NULL if in command line)
            const_cast<char *>(
                command.c_str()), // Command line with executable and arguments
            nullptr,              // Process handle not inheritable
            nullptr,              // Thread handle not inheritable
            false,                // Handle inheritance option
            0,                    // Creation flags (e.g., CREATE_NEW_CONSOLE)
            nullptr,              // Use parent's environment block
            nullptr,              // Use parent's starting directory
            &si,                  // Pointer to STARTUPINFO structure
            &pi                   // Pointer to PROCESS_INFORMATION structure
        );

        if (!success) {
          DD_THROW(OpsFusion::dd_format(
              "Failed to create process. Error code: {}", GetLastError()));
        } else {
          RYZENAI_LOG_TRACE("Process created successfully! ");
        }
        DWORD waitResult = WaitForSingleObject(pi.hProcess, INFINITE);
        if (waitResult == WAIT_OBJECT_0) {
          RYZENAI_LOG_TRACE("Child process completed successfully");
        } else {
          DD_THROW(OpsFusion::dd_format(
              "WaitForSingleObject failed with error code: {}",
              GetLastError()));
        }
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
      }
      xrt::elf elf{elf_path};
      subgraph_elfs_mod_.emplace_back(xrt::module({elf}));
    } catch (std::exception e) {
      DD_THROW(OpsFusion::dd_format("convert_to_elf failed with error: {}",
                                    e.what()));
      return subgraph_elfs_mod_;
    }
  }
#else
  DD_THROW("Preemption currently supported on WINDOW OS");
#endif
  return subgraph_elfs_mod_;
}

void FusionRuntime::init(const Metadata &meta, const std::string &base_dir,
                         const DDConfig &cfg) {

  cfg_.cache_dir = cfg.cache_dir;
  cfg_.profile = cfg.profile;
  cfg_.model_name = cfg.model_name;
  OpInterface::set_dd_base_dir(base_dir);

  // load_state();

  initialize_runtime();
  use_xclbin_parse_data_ = parse_xclbin_metadata(op_pdi_map_, cfg_.model_name);
  // Fused transaction vectors are in partition order.
  fused_instr_vec_ = generate_fused_txns(meta_);
  convert_to_elf(txns_, meta_, ctrl_pkt_vec_file_ptr_);
  initialize_kernels(meta_, kernel_name_to_pdi_idx_);
  reallocate_xrt_bos(meta_, cfg.use_lazy_scratch_bo);
  host_to_dev_memcpy(cfg.use_lazy_scratch_bo);
  if (!elf_flow_ && !cfg.use_lazy_scratch_bo) {
    patch_ctrl_pkt(meta_);
  }
  prepare_formatting_ops();

  // MA TODO: one transaction binary can have multiple kernels (elf + non elf)
  {
    // this block determines if we should either use "heap" or "stack" for
    // instruction BO since this a global state, add lock guard e.g. trying to
    // read map but another thread does an insertion which cause reallocation
    std::lock_guard<std::mutex> guard(instr_state_mutex);
    bool repartition_instr =
        check_context_instr_size(fused_instr_vec_, INSTR_XRT_BO_HEAP_SIZE);

    bool need_realloc = false;

    do {
      repartition_instr = repartition_instr || need_realloc;

      use_instr_sw_cache_ = repartition_instr;

      while (repartition_instr) {
        // this is to ensure current set of txn binaries fit into
        // the static instr BO's
        bool split = split_max_partition_pass(meta_, fused_instr_vec_,
                                              INSTR_BUFFER_SIZE);
        DD_THROW_IF(!split,
                    OpsFusion::dd_format("Instruction partition failed!"));
        fused_instr_vec_ = generate_fused_txns(meta_);
        repartition_instr =
            check_partition_instr_size(fused_instr_vec_, INSTR_BUFFER_SIZE);
      }
      need_realloc = allocate_instr_bos(fused_instr_vec_);
    } while (need_realloc);
  }

  if (use_instr_sw_cache_) {
    convert_to_elf(txns_, meta_, ctrl_pkt_vec_file_ptr_);
    initialize_kernels(meta_, kernel_name_to_pdi_idx_);
  }
  if (!elf_flow_) {
    populate_instr_bos(fused_instr_vec_);
    if (!cfg.use_lazy_scratch_bo) {
      setup_xrt_run(meta_);
    }
  } else {
    setup_xrt_run_elf(meta_);
  }
  release_host_resources();
  RYZENAI_LOG_TRACE("FusionRuntime : Init ... DONE");
}

// For every Op, collect all the const data it has.
// Pass everything to the OpInterface and let it copy to
//    the right place.
void FusionRuntime::load_const(const Metadata &meta,
                               std::map<std::string, SimpleSpan> &const_map) {
  RYZENAI_LOG_TRACE("FusionRuntime : Load const ...");

  for (const auto &op_info : meta.op_list) {
    const auto &tensor_names = op_info.const_args;

    std::vector<Tensor> const_tensors;
    // Read const inputs from files only if tensor_names is not empty.
    // initialize_const_params() is called regardless for each op later.
    // This enabled operators to copy LUTs / other data to AIE. This is
    // required for operators like bf16 Silu/Gelu when ONNX op does not have
    // constant input.
    if (!tensor_names.empty()) {
      for (const auto &name : tensor_names) {
        const auto &tensor_info = meta.tensor_map.at(name);
        auto *const_ptr = MAP_AT(const_map, name).loc;
        const_tensors.push_back(
            {const_ptr, tensor_info.shape, tensor_info.dtype});
      }
    }

    // Get the offset of this op's const buffer in const bo.
    size_t offset = 0;
    // if const_map is empty, call all initilize_const_params for constant
    // initilization, if any.
    if (meta.const_map.find(op_info.name) != meta.const_map.end()) {
      const auto &tensor_info = meta.const_map.at(op_info.name);
      offset = tensor_info.offset;
    }

    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    using signature = void(ConstBufferIO &, const std::vector<Tensor> &,
                           const std::map<std::string, std::any> &);
    fseek64(const_vec_file_ptr_, offset, SEEK_SET);
    auto io = TmpFileConst(const_vec_file_ptr_);
    DD_INVOKE_OVERLOADED_OPMETHOD(initialize_const_params, signature, op.get(),
                                  op_info, io, const_tensors, op_info.attr);
  }
  fseek64(const_vec_file_ptr_, 0, SEEK_SET);
  const_vec_file_md5_ = calculate_md5sum(const_vec_file_ptr_);
  RYZENAI_LOG_TRACE("FusionRuntime : Load const ... DONE");
}

void FusionRuntime::fill_super_instr(
    const Metadata &meta, std::map<std::string, SimpleSpan> &const_map) {
  RYZENAI_LOG_TRACE("FusionRuntime : Fill Super Instrns ... ");
  for (const auto &op_info : meta.op_list) {
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    auto offset = MAP_AT(meta.super_instr_map, op_info.name).offset;

    std::map<std::string, void *> const_buf_ptrs;
    for (const auto &name : op_info.const_args) {
      const_buf_ptrs[name] = MAP_AT(const_map, name).loc;
    }
    std::vector<Tensor> tensors =
        MetaUtils::collect_op_tensors(meta, op_info, const_buf_ptrs);

    auto super_instr =
        DD_INVOKE_OPMETHOD(get_super_kernel_params, op.get(), op_info, tensors,
                           tensors, op_info.attr);
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Copying super instr to bo : offset:{}, size:{}",
                             offset, super_instr.size()));
    fseek64(super_instr_vec_file_ptr_, offset, SEEK_SET);
    std::fwrite(super_instr.data(), 1, super_instr.size(),
                super_instr_vec_file_ptr_);
  }
  fseek64(super_instr_vec_file_ptr_, 0, SEEK_SET);

  RYZENAI_LOG_TRACE("FusionRuntime : Fill Super Instrns ... DONE");
}

void FusionRuntime::fill_ctrl_pkts(const Metadata &meta) {
  RYZENAI_LOG_TRACE("FusionRuntime : Fill Control Packets ... ");
  for (const auto &op_info : meta.op_list) {
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    auto offset = MAP_AT(meta.ctrl_pkt_map, op_info.name).offset;

    std::vector<Tensor> tensors = MetaUtils::collect_op_tensors(meta, op_info);

    auto ctrl_pkts = DD_INVOKE_OPMETHOD(get_ctrl_pkts, op.get(), op_info,
                                        tensors, tensors, op_info.attr);
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Copying ctrl pkts to bo : offset:{}, size:{}",
                             offset, ctrl_pkts.size()));
    fseek64(ctrl_pkt_vec_file_ptr_, offset, SEEK_SET);
    std::fwrite(ctrl_pkts.data(), 1, ctrl_pkts.size(), ctrl_pkt_vec_file_ptr_);
  }
  fseek64(ctrl_pkt_vec_file_ptr_, 0, SEEK_SET);

  RYZENAI_LOG_TRACE("FusionRuntime : Fill Control Packets ... DONE");
}

static size_t read_file_to_bo(xrt::bo &bo, FILE *file, size_t seek) {
  constexpr size_t buffer_size = 4096;
  char buffer[buffer_size];
  size_t read = 0;
  while ((read = fread(buffer, 1, buffer_size, file)) > 0) {
    bo.write(buffer, read, seek);
    seek += read;
  }
  return seek;
}

void FusionRuntime::host_to_dev_memcpy(bool use_lazy_scratch_bo) {
  {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : try to initialize const bo, "
        "curr_size:{}, md5: {}, use_count:{}",
        const_bo_sz_, const_vec_file_md5_, const_bo_.use_count()));
    auto lock_for_const_bo = acquire_lock_for_const_bo();
    const char *data =
        static_cast<const char *>(const_bo_->map()); // Explicit cast
    bool is_uninitialized =
        std::all_of(data, data + const_bo_->size(),
                    [](char c) { return c == XRT_BO_INIT_VALUE; });
    if (is_uninitialized) {
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "FusionRuntime : do initializing const bo, "
          "curr_size:{}, md5: {}, use_count:{}",
          const_bo_sz_, const_vec_file_md5_, const_bo_.use_count()));
      std::ignore = read_file_to_bo(*const_bo_, const_vec_file_ptr_, 0);
    } else {
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "FusionRuntime : cancel initializing const bo, it was initalized by "
          "other "
          "curr_size:{}, md5: {}, use_count:{}",
          const_bo_sz_, const_vec_file_md5_, const_bo_.use_count()));
    }
  }
  std::ignore = read_file_to_bo(input_bo_, input_vec_file_ptr_, 0);
  size_t super_instr_vec_size =
      read_file_to_bo(super_instr_bo_, super_instr_vec_file_ptr_, 0);
  if (ctrl_pkt_vec_file_ptr_ && !elf_flow_) {
    std::ignore = read_file_to_bo(super_instr_bo_, ctrl_pkt_vec_file_ptr_,
                                  super_instr_vec_size);
  }
  output_bo_.write(output_vec_.data(), output_vec_.size(), 0);

  const_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  output_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  super_instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (!use_lazy_scratch_bo) {
    scratch_bo_.write(scratch_vec_.data(), scratch_vec_.size(), 0);
    scratch_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
}

void FusionRuntime::setup_xrt_run(const Metadata &meta) {
  // IMPORTANT: this should only be called after instruction
  //            and data BO's have been allocated
  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ...");
  for (int partition_idx = 0; partition_idx < meta.partitions.size();
       partition_idx++) {
    if (!cfg_.txn_opt) {
      runs_[partition_idx].set_arg(0, std::uint64_t{2});
    } else {
      runs_[partition_idx].set_arg(0, std::uint64_t{3});
    }

    // instruction BO and instruction size will be updated on the fly
    // since same PDI could be running different potions of transaction binary

    runs_[partition_idx].set_arg(3, input_bo_.address() + DDR_AIE_ADDR_OFFSET);
    runs_[partition_idx].set_arg(4, output_bo_.address() + DDR_AIE_ADDR_OFFSET);
    runs_[partition_idx].set_arg(5,
                                 scratch_bo_.address() + DDR_AIE_ADDR_OFFSET);
    runs_[partition_idx].set_arg(6, const_bo_->address() + DDR_AIE_ADDR_OFFSET);
    runs_[partition_idx].set_arg(7, super_instr_bo_.address() +
                                        DDR_AIE_ADDR_OFFSET);
  }

  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ... DONE");
}

void FusionRuntime::setup_xrt_run_elf(const Metadata &meta) {
  // IMPORTANT: this should only be called after instruction
  //            and data BO's have been allocated
  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT_ELF Run objects ...");
  for (int i = 0; i < meta.partitions.size(); i++) {

    auto &run = runs_[i];
    run.set_arg(0, ELF_OPCODE);
    run.set_arg(1, 0);
    run.set_arg(2, 0);
    run.set_arg(3, input_bo_);
    run.set_arg(4, output_bo_);
    run.set_arg(5, scratch_bo_);
    run.set_arg(6, *const_bo_);
    run.set_arg(7, super_instr_bo_);
    // run.set_arg(8, 0);
  }

  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ... DONE");
}

std::vector<std::vector<uint8_t>>
FusionRuntime::generate_fused_txns(const Metadata &meta) {

  std::vector<std::vector<uint8_t>> fused_txns;
  fused_txns.reserve(meta.partitions.size());

  txns_.clear();
  txns_.reserve(meta.partitions.size());

  size_t partition_index = 0;
  for (const auto &partition : meta.partitions) {
    // get fused transactions
    std::vector<txn_vec_t> txn_vecs;
    auto &op_range = partition.op_range;
    size_t const num_ops = op_range.second > op_range.first
                               ? (op_range.second - op_range.first)
                               : (1);
    txn_vecs.reserve(num_ops);
    for (auto ind = op_range.first; ind < op_range.second; ind++) {
      const auto &op_info = meta.op_list.at(ind);
      auto txn_bin = op_info.txn_bin;
      txn_vecs.push_back(txn_bin);
    }
    auto fused_txn = utils::txn_util::fuse_txns(txn_vecs);
    if (cfg_.txn_opt) {
      utils::txn_util txn_util_instance;
      auto new_fused_txn = txn_util_instance.convert_to_opt_txn(fused_txn);
      txns_.push_back(new_fused_txn);
    } else {
      txns_.push_back(fused_txn);
    }
    utils::txn_util txn = utils::txn_util(txns_.back());
    auto ibuf_op = transaction_op(txn.txn);
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Partition {} Fused Txn Summary :\n{}",
                             partition_index, txn.summarize()));

    fused_txns.push_back(std::move(ibuf_op.get_txn_op()));

    partition_index += 1;
  }

  RYZENAI_LOG_TRACE("FusionRuntime : Generate Fused Transactions ... DONE");

  return fused_txns;
}

bool FusionRuntime::check_context_instr_size(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec,
    const size_t limit) {

  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);
  size_t curr_instr_size = xrt_instr_state.at(handle).heap_total_size;
  size_t instr_size = 0;

  for (const auto &instr : fused_instr_vec) {
    instr_size += instr.size();
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "FusionRuntime : check_context_instr_size requesting "
      "instruction BO instr_size: {}, "
      "curr_instr_size: {}, limit: {}",
      instr_size, curr_instr_size, limit));

  return ((curr_instr_size + instr_size) > limit);
}

bool FusionRuntime::check_partition_instr_size(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec,
    const size_t partition_limit) {

  bool repartition = false;
  size_t instruction_idx = 0;

  for (const auto &instr : fused_instr_vec) {
    auto instr_size = instr.size();
    if (instr_size > partition_limit) {
      repartition = true;
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "FusionRuntime : Need to repartition instruction: {}, size: {}",
          instruction_idx, instr_size));
    }

    instruction_idx++;
  }

  return repartition;
}

bool FusionRuntime::allocate_instr_bos(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec) {

  // TODO: move this deallocation to beginning of init??
  // Added here since instr_bos get cleared
  xrt_core::hwctx_handle *handle = static_cast<xrt_core::hwctx_handle *>(ctx_);

  for (auto &instr_bo : instr_bos_) {
    xrt_instr_state.at(handle).heap_total_size -=
        Utils::align_to_next(instr_bo.size(), INSTR_XRT_BO_ALIGNMENT);
  }
  xrt_instr_state.at(handle).num_instr_bos -= instr_bos_.size();

  instr_bos_.clear();

  if (use_instr_sw_cache_) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("FusionRuntime : skip allocate instr_bo"));
    return false;
  }

  instr_bos_.reserve(fused_instr_vec.size());

  bool alloc_failed = false;

  for (const auto &instr : fused_instr_vec) {
    size_t instr_size = instr.size();
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Reallocating instr_bo, new_size:{}", instr_size));
    try {
      instr_bos_.emplace_back(xrt::bo(ctx_, instr_size,
                                      xrt::bo::flags::cacheable,
                                      kernels_[0].group_id(1)));
    } catch (...) {
      RYZENAI_LOG_TRACE(
          OpsFusion::dd_format("FusionRuntime : Reallocating instr BO failed! "
                               "Fallback to static buffers"));
      alloc_failed = true;
      break;
    }
  }

  if (alloc_failed) {
    // For a particular FusionRuntime object, its instruction BOs
    // wiil either be all in "heap" or "stack"
    // in this case, fall back to using the "stack" instruction BOs
    instr_bos_.clear();
    return true;
  }

  // update state if allocation was successful
  for (const auto &instr : fused_instr_vec) {
    size_t instr_size = instr.size();
    xrt_instr_state.at(handle).heap_total_size +=
        Utils::align_to_next(instr_size, INSTR_XRT_BO_ALIGNMENT);
  }
  xrt_instr_state.at(handle).num_instr_bos += fused_instr_vec.size();
  return false;
}

void FusionRuntime::populate_instr_bos(
    const std::vector<std::vector<uint8_t>> &fused_instr_vec) {

  if (use_instr_sw_cache_) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("FusionRuntime : skip populate instr_bo"));
    return;
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("FusionRuntime : populate instr_bo"));

  size_t instr_index = 0;
  for (const auto &instr : fused_instr_vec) {
    write_to_bo(instr_bos_.at(instr_index), 0, /*offset*/
                fused_instr_vec.at(instr_index).data(),
                fused_instr_vec.at(instr_index).size());

    instr_index += 1;
  }
}

void FusionRuntime::allocate_host_bos(const Metadata &meta) {
  RYZENAI_LOG_TRACE("Allocating Data Buffers ...");
  const_vec_file_ptr_ = Utils::create_tmpfile();
  input_vec_file_ptr_ = Utils::create_tmpfile();
  scratch_vec_ =
      std::vector<uint8_t>(MAP_AT(meta.fused_tensors, "scratch").size);
  output_vec_ = std::vector<uint8_t>(MAP_AT(meta.fused_tensors, "out").size);
  super_instr_vec_file_ptr_ = Utils::create_tmpfile();
  ctrl_pkt_vec_file_ptr_ = Utils::create_tmpfile();
  RYZENAI_LOG_TRACE("Allocating Data Buffers ... DONE");
}

// TODO : Calling .size() on empty xrt::bo crashes.
void FusionRuntime::reallocate_xrt_bos(const Metadata &meta,
                                       bool use_lazy_scratch_bo) {
  RYZENAI_LOG_TRACE("Reallocating Data BOs ...");
  size_t new_size =
      std::max(MAP_AT(meta.fused_tensors, "super_instr").size, XRT_BO_MIN_SIZE);
  if ((meta.major_version >= 1) && (meta.minor_version >= 1)) {
    if (!elf_flow_) {
      new_size = std::max(MAP_AT(meta.fused_tensors, "super_instr").size +
                              MAP_AT(meta.fused_tensors, "ctrl_pkt").size,
                          XRT_BO_MIN_SIZE);
    } else {
      new_size = std::max(MAP_AT(meta.fused_tensors, "super_instr").size,
                          XRT_BO_MIN_SIZE);
    }

    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Updated super_instr_bo_size_ : {}", new_size));
  }

  if (super_instr_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Reallocating input bo, curr_size:{}, new_size:{}",
        super_instr_bo_sz_, new_size));
    super_instr_bo_ =
        allocate_xrt_buffer(ctx_, new_size, xrt::bo::flags::host_only,
                            kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(super_instr_bo_.map(), XRT_BO_INIT_VALUE, super_instr_bo_.size());
    super_instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  super_instr_bo_sz_ = new_size;

  new_size =
      std::max(MAP_AT(meta.fused_tensors, "const").size, XRT_BO_MIN_SIZE);
  if (const_bo_sz_ < new_size) {
    {
      auto lock = acquire_lock_for_const_bo();
      DD_ASSERT(!const_vec_file_md5_.empty(),
                "const file md5 checksum must not be empty");

      if (elf_flow_) {
        const_bo_ = vitis::ai::WeakStore<std::string, xrt::ext::bo>::create(
            const_vec_file_md5_, ctx_, new_size);
      } else {
        const_bo_ = vitis::ai::WeakStore<std::string, xrt::bo>::create(
            const_vec_file_md5_, ctx_, new_size, xrt::bo::flags::host_only,
            kernels_[0].group_id(HOST_BO_GROUP_ID));
      }
      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "FusionRuntime : Reallocating const bo, "
          "curr_size:{}, new_size:{} md5:{} use_count:{}",
          const_bo_sz_, new_size, const_vec_file_md5_, const_bo_.use_count()));
      if (const_bo_.use_count() == 1) {
        memset(const_bo_->map(), XRT_BO_INIT_VALUE, const_bo_->size());
        const_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    }
  }
  const_bo_sz_ = new_size;

  new_size = std::max(MAP_AT(meta.fused_tensors, "in").size, XRT_BO_MIN_SIZE);
  if (input_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Reallocating input bo, curr_size:{}, new_size:{}",
        input_bo_sz_, new_size));
    input_bo_ = allocate_xrt_buffer(ctx_, new_size, xrt::bo::flags::host_only,
                                    kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(input_bo_.map(), XRT_BO_INIT_VALUE, input_bo_.size());
    input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  input_bo_sz_ = new_size;

  new_size = std::max(MAP_AT(meta.fused_tensors, "out").size, XRT_BO_MIN_SIZE);
  if (output_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Reallocating output bo, curr_size:{}, new_size:{}",
        output_bo_sz_, new_size));
    output_bo_ = allocate_xrt_buffer(ctx_, new_size, xrt::bo::flags::host_only,
                                     kernels_[0].group_id(HOST_BO_GROUP_ID));
    memset(output_bo_.map(), XRT_BO_INIT_VALUE, output_bo_.size());
    output_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  output_bo_sz_ = new_size;

  new_size = std::max(MAP_AT(meta.fused_tensors, "scratch").size +
                          meta.max_op_scratch_pad_size,
                      XRT_BO_MIN_SIZE);
  if (scratch_bo_sz_ < new_size) {
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "FusionRuntime : Reallocating scratch bo, curr_size:{}, new_size:{}",
        scratch_bo_sz_, new_size));
    if (!use_lazy_scratch_bo) {
      scratch_bo_ =
          allocate_xrt_buffer(ctx_, new_size, xrt::bo::flags::host_only,
                              kernels_[0].group_id(HOST_BO_GROUP_ID));
      memset(scratch_bo_.map(), XRT_BO_INIT_VALUE, scratch_bo_.size());
      scratch_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    } else {
      scratch_bo_allocate_ = true;
    }
  }
  scratch_bo_sz_ = new_size;

  RYZENAI_LOG_TRACE("Reallocating Data BOs ... DONE");
  RYZENAI_LOG_TRACE(OpsFusion::dd_format(
      "\ninput bo size : {}\noutput bo size : {}\nconst bo size : "
      "{}\nscratch bo size : {}\nsuper_instr_bo size : {}",
      input_bo_sz_, output_bo_sz_, const_bo_sz_, scratch_bo_sz_,
      super_instr_bo_sz_));
}

std::vector<std::vector<uint8_t>> FusionRuntime::get_txns() { return txns_; }

void FusionRuntime::initialize_inputs(const Metadata &meta) {
  auto input_vec = std::vector<uint8_t>(MAP_AT(meta.fused_tensors, "in").size);
  uint8_t *out_ptr = output_vec_.data();
  uint8_t *scratch_ptr = scratch_vec_.data();
  std::array<uint8_t *, 3> buf_ptrs = {input_vec.data(), out_ptr, scratch_ptr};

  for (const auto &op_info : meta.op_list) {
    std::vector<Tensor> tensors;
    auto args = OpsFusion::get_op_args(op_info);
    for (const auto &buf_name : args) {
      const auto &tensor_info = MAP_AT(meta.tensor_map, buf_name);
      const auto &packed_tensor_name = tensor_info.parent_name;

      // TODO : Unsafe check. Fails if buf name changed in future
      if (!(packed_tensor_name == "in" || packed_tensor_name == "scratch")) {
        break;
      }

      const auto buf_arg_id =
          MAP_AT(meta.fused_tensors, packed_tensor_name).arg_idx;
      uint8_t *ptr = buf_ptrs[buf_arg_id] + tensor_info.offset;
      Tensor tensor = {ptr, tensor_info.shape, tensor_info.dtype};
      tensors.push_back(std::move(tensor));
    }
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    DD_INVOKE_OPMETHOD(initialize_inputs, op.get(), op_info, tensors,
                       op_info.attr);
  }
  Utils::dump_to_tmpfile(input_vec_file_ptr_,
                         reinterpret_cast<char *>(input_vec.data()),
                         input_vec.size());
}

void FusionRuntime::merge_inputs(const std::vector<Tensor> &inputs,
                                 const Metadata &meta) {
  RYZENAI_LOG_TRACE("Packing Inputs ... ");
  size_t n_meta_inputs = MetaUtils::get_num_inputs(meta);
  DD_ASSERT(
      inputs.size() == n_meta_inputs,
      OpsFusion::dd_format(
          "Number of inputs ({}) doesn't match with that of metadata ({})",
          inputs.size(), n_meta_inputs));

  auto t1 = GET_ELAPSED_TIME_NS();
  const auto &in_buf_names = MAP_AT(meta.fused_tensors, "in").packed_tensors;
  for (int i = 0; i < in_buf_names.size(); i++) {
    size_t sz = std::accumulate(inputs[i].shape.begin(), inputs[i].shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(inputs[i].dtype);

    // TODO : Check if input buffer size matches with metadata
    const auto &tensor_info = MAP_AT(meta.tensor_map, in_buf_names[i]);
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "copying input:{} to input bo at offset:{} and size:{}",
        in_buf_names[i], tensor_info.offset, sz));
    char *inp_ptr = (char *)inputs[i].data;
    input_bo_.write(inputs[i].data, sz, tensor_info.offset);
  }
  auto t2 = GET_ELAPSED_TIME_NS();
  input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto t3 = GET_ELAPSED_TIME_NS();

  input_copy_time_ = static_cast<int64_t>(t2 - t1);
  input_sync_time_ = static_cast<int64_t>(t3 - t2);
  RYZENAI_LOG_TRACE("Packing Inputs ... DONE");
}

void FusionRuntime::split_outputs(const std::vector<Tensor> &outputs,
                                  const Metadata &meta) {
  RYZENAI_LOG_TRACE("Unpacking Outputs ...");
  size_t n_meta_outputs = MetaUtils::get_num_outputs(meta);
  DD_ASSERT(outputs.size() == n_meta_outputs,
            OpsFusion::dd_format(
                "Number of outputs ({}) doesn't match with number of "
                "metadata outputs ({})",
                outputs.size(), n_meta_outputs));

  auto t1 = GET_ELAPSED_TIME_NS();
  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  void *output_bo_ptr = output_bo_.map();
  auto t2 = GET_ELAPSED_TIME_NS();

  const auto &out_buf_names = meta.fused_tensors.at("out").packed_tensors;
  auto hwout_tensors = MetaUtils::get_output_tensors(meta);
  for (int i = 0; i < out_buf_names.size(); i++) {
    DD_ASSERT(outputs[i].shape == hwout_tensors[i].shape,
              dd_format("output tensor shapes doesn't match with the "
                        "Runtime output tensor shapes"));
    DD_ASSERT(outputs[i].dtype == hwout_tensors[i].dtype,
              dd_format("output tensor dtype doesn't match with the "
                        "Runtime output tensor dtype"));

    const auto &tensor_info = MAP_AT(meta.tensor_map, out_buf_names[i]);
    size_t sz =
        std::accumulate(outputs[i].shape.begin(), outputs[i].shape.end(),
                        size_t{1}, std::multiplies{}) *
        Utils::get_size_of_type(outputs[i].dtype);

    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "Reading output:{} from output bo at offset:{} and size:{}",
        out_buf_names[i], tensor_info.offset, sz));

    // TODO : Do the depad here
    hwout_tensors[i].data = (char *)output_bo_ptr + tensor_info.offset;

    // copy_data(hwout_tensors[i], outputs[i]);
    size_t tensor_bo_sz = tensor_info.size_in_bytes;
    auto &[op_info, op] = producer_ops_[i];

    // format output is not handled for more than one output
    // log a warning and skip format_output_api for now.
    // TODO: Can we call this for multiple outputs in a loop?
    if (op_info.out_args.size() == 1) {
      DD_INVOKE_OPMETHOD(format_output, op.get(), op_info, outputs[i],
                         hwout_tensors[i].data, tensor_bo_sz, /* index */ 0,
                         op_info.attr);
    } else {
      RYZENAI_LOG_TRACE(
          "WARNING: format_output_api not handled for more than one output");

      auto out_tensor_sz =
          std::accumulate(outputs[i].shape.begin(), outputs[i].shape.end(),
                          size_t{1}, std::multiplies{}) *
          Utils::get_size_of_type(outputs[i].dtype);
      memcpy(outputs[i].data, hwout_tensors[i].data, tensor_bo_sz);
    }
  }
  auto t3 = GET_ELAPSED_TIME_NS();

  output_copy_time_ = static_cast<int64_t>(t3 - t2);
  output_sync_time_ = static_cast<int64_t>(t2 - t1);
  RYZENAI_LOG_TRACE("Unpacking Outputs ... DONE");
}

// FIXME : Currently Metadata doesn't differentiate between input and output
// args of a node. It is just one large combined vector. This makes it
// difficult to find the producer of a particular buffer. SO below
// implimentation works only if node has only one output.
void FusionRuntime::prepare_formatting_ops() {
  // TODO : Do for input tensors as well in future.

  // Output Tensors
  std::vector<
      std::pair<OpsFusion::Metadata::OpInfo, std::unique_ptr<OpInterface>>>
      producer_ops;

  producer_ops.reserve(
      MAP_AT(meta_.fused_tensors, "out").packed_tensors.size());
  for (const auto tensor_name :
       MAP_AT(meta_.fused_tensors, "out").packed_tensors) {
    for (const auto &op_info : meta_.op_list) {
      auto args = OpsFusion::get_op_args(op_info);
      if (!args.empty() && args.back() == tensor_name) {
        auto op = OpBuilder::create(op_info.name, op_info, meta_.tensor_map);
        producer_ops.emplace_back(op_info, std::move(op));
        break;
      }
    }
  }
  producer_ops_ = std::move(producer_ops);
}

const Metadata &FusionRuntime::get_meta() const { return meta_; }

std::map<std::string, std::vector<uint8_t>>
FusionRuntime::unpack_internal_buffers(const std::string &dir) {
  std::map<std::string, std::vector<uint8_t>> res;
  auto meta = meta_;

  auto unpack_buffer = [&meta, &res](const std::string &fused_buffer,
                                     xrt::bo &bo) {
    const auto &in_buf_names =
        MAP_AT(meta.fused_tensors, fused_buffer).packed_tensors;
    for (const auto &buf_name : in_buf_names) {
      const auto &tensor_info = MAP_AT(meta.tensor_map, buf_name);
      auto sz = tensor_info.size_in_bytes;
      auto offset = tensor_info.offset;

      RYZENAI_LOG_TRACE(OpsFusion::dd_format(
          "Unpacking input:{} from input bo at offset:{} and size:{}", buf_name,
          offset, sz));

      std::vector<uint8_t> vec(sz);
      bo.read(vec.data(), sz, tensor_info.offset);
      res[buf_name] = std::move(vec);
    }
  };

  input_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const_bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  scratch_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  super_instr_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Not a deadcode. Keeping it as a reminder so it won't be repeated
  // "Don't sync instr bo from device, it will hang the board."

  unpack_buffer("in", input_bo_);
  unpack_buffer("scratch", scratch_bo_);
  unpack_buffer("out", output_bo_);

  if (!dir.empty()) {
    std::cout << "[WARNING] Writing DD intrenal buffers to " << dir
              << std::endl;
    std::cout << "[WARNING] This incurs performance overhead. Set env variable "
                 "DD_WRITE_INTERNAL_BUFS=0 to disable it."
              << std::endl;

    std::filesystem::path dir_path{dir};
    std::filesystem::create_directories(dir_path);
    for (const auto &[name, data] : res) {
      auto newname = replace_characters(name, "/:,", '_');
      auto filename = OpsFusion::dd_format("{}/{}.bin", dir, newname);
      std::ofstream ofs(filename, std::ios::binary);
      DD_ASSERT(ofs,
                OpsFusion::dd_format("Couldn't open {} for writing", filename))
      ofs.write((char *)data.data(), data.size());
      ofs.close();
    }

    auto hash_filename = dir_path / "buffer_hash.txt"s;
    std::ofstream hash_fs(hash_filename);

    hash_fs << "------------------------------------" << std::endl;
    hash_fs << "ACTIVATIONS (TensorName, Hash)" << std::endl;
    hash_fs << "------------------------------------" << std::endl;
    hash_fs << std::endl;
    for (const auto &name : meta.fused_tensors.at("in").packed_tensors) {
      const auto &data = MAP_AT(res, name);
      hash_fs << name << ", input, " << compute_hash(data.data(), data.size())
              << std::endl;
    }
    for (const auto &op_info : meta.op_list) {
      for (const auto &name : op_info.out_args) {
        const auto &data = MAP_AT(res, name);
        hash_fs << name << ", " << op_info.type << ", "
                << compute_hash(data.data(), data.size()) << std::endl;
      }
    }

    hash_fs << std::endl;
    hash_fs << "------------------------------------" << std::endl;
    hash_fs << "CONSTANT DATA (OpName, OpType, Hash)" << std::endl;
    hash_fs << "------------------------------------" << std::endl;
    hash_fs << std::endl;

    auto *const_bo_ptr = const_bo_->map<int8_t *>();
    for (const auto &op : meta_.op_list) {
      if (meta_.const_map.find(op.name) == meta_.const_map.end()) {
        continue;
      }
      const auto &const_span = MAP_AT(meta_.const_map, op.name);
      hash_fs << op.name << ", " << op.type << ", "
              << compute_hash(const_bo_ptr + const_span.offset, const_span.size)
              << std::endl;

      auto newname = replace_characters(op.name, "/:,", '_');
      auto filename = OpsFusion::dd_format("{}/{}.const", dir, newname);
      std::ofstream ofs(filename, std::ios::binary);
      DD_ASSERT(ofs,
                OpsFusion::dd_format("Couldn't open {} for writing", filename))
      ofs.write((char *)const_bo_ptr + const_span.offset, const_span.size);
      ofs.close();
    }

    // Hash for full bos.
    hash_fs << std::endl;
    hash_fs << "------------------------------------" << std::endl;
    hash_fs << "FULL BO (BOName, Hash)" << std::endl;
    hash_fs << "------------------------------------" << std::endl;
    hash_fs << std::endl;

    hash_fs << "input_bo, " << compute_hash(input_bo_.map(), input_bo_.size())
            << std::endl;
    hash_fs << "output_bo, "
            << compute_hash(output_bo_.map(), output_bo_.size()) << std::endl;
    hash_fs << "scratch_bo, "
            << compute_hash(scratch_bo_.map(), scratch_bo_.size()) << std::endl;
    hash_fs << "const_bo, " << compute_hash(const_bo_->map(), const_bo_->size())
            << std::endl;
    hash_fs << "super_kernel_bo, "
            << compute_hash(super_instr_bo_.map(), super_instr_bo_.size())
            << std::endl;
    for (size_t i = 0; i < instr_bos_.size(); ++i) {
      auto &instr_bo = ARRAY_AT(instr_bos_, i);
      hash_fs << "instruction_bo:" << i << ", "
              << compute_hash(instr_bo.map(), instr_bo.size()) << std::endl;
    }
  }

  return res;
}

bool FusionRuntime::parse_xclbin_metadata(OpPDIMap &op_pdi_map,
                                          const std::string &model_name) {
  auto xclbin = ctx_.get_xclbin();
  return parse_xclbin_metadata(op_pdi_map, model_name, xclbin);
}

bool FusionRuntime::parse_xclbin_metadata(
    OpPDIMap &op_pdi_map, const std::string &model_name,
    const std::vector<char> *xclbin_content) {
  auto xclbin = xrt::xclbin(*xclbin_content);

  const auto kernel_names = filter_xrt_kernels(xclbin);

  create_kernel_name_to_pdi_map(kernel_names, kernel_name_to_pdi_idx_);

  return parse_xclbin_metadata(op_pdi_map, model_name, xclbin);
}

bool FusionRuntime::parse_xclbin_metadata(OpPDIMap &op_pdi_map,
                                          const std::string &model_name,
                                          const xrt::xclbin &xclbin) {

  op_pdi_map.op_to_pdi_id_map.clear();
  op_pdi_map.pdi_id_to_kernel_map.clear();

  if (model_name == "") {
    return false;
  }

  try {
    auto vend_meta = ::xclbin::get_axlf_section(
        xclbin.get_axlf(), axlf_section_kind::VENDER_METADATA);
    if (nullptr == vend_meta) {
      // std::cout << "No section found!"<< std::endl;
      return false;
    }

    // std::cout << "Section name... " << vend_meta->m_sectionName <<
    // std::endl; std::cout << "section offset... " <<
    // vend_meta->m_sectionOffset << std::endl; std::cout << "section size...
    // " << vend_meta->m_sectionSize << std::endl;

    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "parse_xclbin_metadata::model_name {}, section offset {} and size {}",
        model_name, vend_meta->m_sectionOffset, vend_meta->m_sectionSize));

    const char *p =
        (const char *)xclbin.get_axlf() + vend_meta->m_sectionOffset;
    const struct vender_metadata *vder = (const struct vender_metadata *)p;
    const char *vender_metadata_p = p + vder->m_image_offset;
    size_t vender_metadata_size = vder->m_image_size;
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("parse_xclbin_metadata::vender_metadata_size {}",
                             vender_metadata_size));

    if (vender_metadata_size == 0) {
      // std::cout << "Empty section!"<< std::endl;
      return false;
    }
    std::vector<std::uint8_t> json_blob(vender_metadata_size);
    // TODO: check how to get section data
    memcpy(json_blob.data(), vender_metadata_p, vender_metadata_size);

    json data;
    try {
      data = json::parse(json_blob.begin(), json_blob.end(), nullptr, true);
    } catch (std::exception &e) {
      // std::cout << e.what() << std::endl;
      DD_THROW(OpsFusion::dd_format("Failed to parse JSON: {} (Detail: {})",
                                    model_name, e.what()));
    }

    for (const auto &[op_type, pdi] : data.at(model_name).items()) {
      if ("DD PDI Metadata version" == op_type) {
        continue;
      }
      std::string pdi_name = pdi.template get<std::string>();
      // in this case, pdi_id == pdi_idx
      if (kernel_name_to_pdi_idx_.find(pdi_name) ==
          kernel_name_to_pdi_idx_.end()) {
        // We may found elf version of the pdi_name.
        continue;
      }
      auto pdi_id = kernel_name_to_pdi_idx_.at(pdi_name);
      op_pdi_map.op_to_pdi_id_map[op_type] = pdi_id;
      op_pdi_map.pdi_id_to_kernel_map[pdi_id] = pdi_name;
    }

    return true;
  } catch (...) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("parse_xclbin_metadata::failed to parse data"));
    // std::cout << "Exception when trying to query section data!" <<
    // std::endl;
    return false;
  }
}

std::once_flag FusionRuntime::logger_flag_;

FusionSubgraphRuntime::FusionSubgraphRuntime(
    const std::string &xclbin, const std::string &kernel_name,
    std::uint32_t num_parallel_contexts)
    : num_contexts_(num_parallel_contexts) {

  DD_THROW_IF(0 == num_parallel_contexts, "Expect at least 1 context");
  ctxs_.reserve(num_parallel_contexts);
  for (std::uint32_t context_id = 0; context_id < num_parallel_contexts;
       context_id++) {
    ctxs_.push_back(
        ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin, context_id)
            ->get_context());
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("FusionSubgraphRuntime : Creating kernel object "
                             ": {} for context_id : {}",
                             kernel_name, context_id));
    xrt::kernel k(ctxs_.back(), kernel_name);
    kernels_.push_back(k);
#ifdef XRT_RUNLIST_EN
    runlists_.emplace_back(ctxs_.back());
#endif
  }

  // make inserting into global state here thread safe
  std::lock_guard<std::mutex> guard(instr_state_mutex);

  for (std::uint32_t context_id = 0; context_id < num_parallel_contexts;
       context_id++) {
    xrt_core::hwctx_handle *handle =
        static_cast<xrt_core::hwctx_handle *>(ctxs_.at(context_id));

    if (xrt_instr_state.find(handle) == xrt_instr_state.end()) {
      xrt_instr_state[handle] = XRTBufferState{};
      for (size_t i = 0; i < NUM_STATIC_INSTR_BUFFERS; i++) {
        xrt_instr_state.at(handle).static_instr_bos.emplace_back(xrt::bo(
            ctxs_.at(context_id), INSTR_BUFFER_SIZE, xrt::bo::flags::cacheable,
            kernels_.at(context_id).group_id(1)));
        xrt_instr_state.at(handle).static_instr_sizes.push_back(0);
      }
      xrt_instr_state.at(handle).num_instr_bos = 2;
      xrt_instr_state.at(handle).num_fusionrt_instances++;
    }
  }
}

FusionSubgraphRuntime::FusionSubgraphRuntime(std::vector<xrt::hw_context> &ctxs,
                                             const std::string &kernel_name)
    : num_contexts_(ctxs.size()), ctxs_(ctxs) {

  DD_THROW_IF(0 == ctxs.size(), "Expect at least 1 context");
  for (std::uint32_t context_id = 0; context_id < num_contexts_; context_id++) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("FusionSubgraphRuntime : Creating kernel object "
                             ": {} for context_id : {}",
                             kernel_name, context_id));
    xrt::kernel k(ctxs_.at(context_id), kernel_name);
    kernels_.push_back(k);
#ifdef XRT_RUNLIST_EN
    runlists_.emplace_back(ctxs_.at(context_id));
#endif
  }

  // make inserting into global state here thread safe
  std::lock_guard<std::mutex> guard(instr_state_mutex);

  for (std::uint32_t context_id = 0; context_id < num_contexts_; context_id++) {
    xrt_core::hwctx_handle *handle =
        static_cast<xrt_core::hwctx_handle *>(ctxs_.at(context_id));

    if (xrt_instr_state.find(handle) == xrt_instr_state.end()) {
      xrt_instr_state[handle] = XRTBufferState{};
      for (size_t i = 0; i < NUM_STATIC_INSTR_BUFFERS; i++) {
        xrt_instr_state.at(handle).static_instr_bos.emplace_back(xrt::bo(
            ctxs_.at(context_id), INSTR_BUFFER_SIZE, xrt::bo::flags::cacheable,
            kernels_.at(context_id).group_id(1)));
        xrt_instr_state.at(handle).static_instr_sizes.push_back(0);
      }
      xrt_instr_state.at(handle).num_instr_bos = 2;
      xrt_instr_state.at(handle).num_fusionrt_instances++;
    }
  }
}

FusionSubgraphRuntime::~FusionSubgraphRuntime() {

  // make book-keeping here thread safe
  // heap_total_size is used to determine if we should
  // use static instruction BO or not
  std::lock_guard<std::mutex> guard(instr_state_mutex);

  for (std::uint32_t context_id = 0; context_id < num_contexts_; context_id++) {
    xrt_core::hwctx_handle *handle =
        static_cast<xrt_core::hwctx_handle *>(ctxs_.at(context_id));

    xrt_instr_state.at(handle).heap_total_size -= Utils::align_to_next(
        instr_bos_.at(context_id).size(), INSTR_XRT_BO_ALIGNMENT);
    // for now only 1 instr BO in subgraph, could need splitting later
    xrt_instr_state.at(handle).num_instr_bos -= 1;
    xrt_instr_state.at(handle).num_fusionrt_instances--;

    if (xrt_instr_state.at(handle).num_fusionrt_instances == 0) {
      xrt_instr_state.erase(handle);
    }
  }
}

void FusionSubgraphRuntime::init(std::vector<Tensor> &tensors,
                                 const std::map<std::string, std::any> attr) {

  auto init_start = GET_ELAPSED_TIME_NS();
  attr_ = attr;

  constexpr bool load_xrt = false;
  auto sg = std::make_unique<ryzenai::xcom::subgraph>(load_xrt);

  // prepare txn bins and instruction BO
  std::vector<uint8_t> txn_bin =
      sg->get_transaction_bin(tensors, tensors, attr);

  bool enable_profile = "" != Utils::get_env_var("DD_ENABLE_PROFILE");
  if (enable_profile) {
    const std::string subgraph_name =
        std::any_cast<std::string>(attr_.find("subgraph_name")->second);
    auto record_time_attrs = get_record_timer_attr(subgraph_name);
    auto record_timer_op = std::make_unique<ryzenai::record_timer>();
    std::vector<uint8_t> record_start_txn_bin =
        record_timer_op->get_transaction_bin(tensors, tensors,
                                             record_time_attrs.first);
    std::vector<uint8_t> record_end_txn_bin =
        record_timer_op->get_transaction_bin(tensors, tensors,
                                             record_time_attrs.second);
    txn_bin = utils::txn_util::fuse_txns(
        {record_start_txn_bin, txn_bin, record_end_txn_bin});
  }
  setup_instr_bo(txn_bin);
  auto buffer_reqs = sg->get_buffer_reqs(tensors, tensors, attr);
  allocate_data_bos(buffer_reqs);
  init_const_inputs(tensors, attr);
  setup_xrt_runlist(attr);

  auto init_end = GET_ELAPSED_TIME_NS();

  auto init_time = (init_end - init_start) / 1000;

  RYZENAI_LOG_INFO("FusionSubgraphRuntime::init time(us)," +
                   std::to_string(init_time) +
                   ",0,0,0"
                   "\n");
}

void FusionSubgraphRuntime::setup_xrt_runlist(
    const std::map<std::string, std::any> &attr) {
  // IMPORTANT: this should only be called after necessary BOs have been
  // allocated
  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ...");

  DD_THROW_IF(attr.find("padded_ifm_tile_offsets") == attr.end(),
              "Missing padded_ifm_tile_offsets attr");
  DD_THROW_IF(attr.find("padded_ofm_tile_shape") == attr.end(),
              "Missing padded_ofm_tile_shape attr");
  DD_THROW_IF(attr.find("num_tiles") == attr.end(), "Missing num_tiles attr");
  DD_THROW_IF(attr.find("out_dtypes") == attr.end(),
              "Can't find out_dtypes attribute for the subgraph");

  const auto padded_ifm_tile_offsets = std::any_cast<std::vector<size_t>>(
      attr.find("padded_ifm_tile_offsets")->second);

  std::vector<size_t> padded_ofm_tile_shape =
      std::any_cast<std::vector<size_t>>(
          attr.find("padded_ofm_tile_shape")->second);

  std::vector<std::string> out_dtypes =
      std::any_cast<std::vector<std::string>>(attr.find("out_dtypes")->second);

  size_t padded_ofm_tile_size = std::accumulate(padded_ofm_tile_shape.begin(),
                                                padded_ofm_tile_shape.end(),
                                                size_t{1}, std::multiplies{}) *
                                Utils::get_size_of_type(out_dtypes.at(0));

  const auto num_tiles = std::any_cast<size_t>(attr.find("num_tiles")->second);

  DD_THROW_IF(num_tiles != padded_ifm_tile_offsets.size(),
              "Inconsistent input num tiles");

  sched_num_tiles_per_context_.clear();
  prefix_sum_sched_num_tiles_.clear();

  if (num_tiles <= num_contexts_) {
    sched_num_tiles_per_context_.reserve(num_tiles);
    for (std::uint32_t context_id = 0; context_id < num_tiles; context_id++) {
      sched_num_tiles_per_context_.push_back(1);
    }
  } else {
    size_t num_tiles_per_context = num_tiles / num_contexts_;
    size_t remaining_tiles = num_tiles - num_tiles_per_context * num_contexts_;

    sched_num_tiles_per_context_.reserve(num_contexts_);

    for (std::uint32_t context_id = 0; context_id < num_contexts_;
         context_id++) {
      sched_num_tiles_per_context_.push_back(num_tiles_per_context);
    }

    for (std::uint32_t context_id = 0; context_id < remaining_tiles;
         context_id++) {
      sched_num_tiles_per_context_.at(context_id) += 1;
    }
  }

  const size_t num_sched_contexts = sched_num_tiles_per_context_.size();
  std::uint32_t global_tile_idx = 0;

  runs_.clear();

  for (std::uint32_t context_id = 0; context_id < num_sched_contexts;
       context_id++) {
#ifdef XRT_RUNLIST_EN
    runlists_.at(context_id).reset();
#endif
    auto num_tiles = sched_num_tiles_per_context_.at(context_id);
    prefix_sum_sched_num_tiles_.push_back(global_tile_idx);
    for (std::uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      runs_.emplace_back(kernels_.at(context_id));
      runs_.back().set_arg(0, OPCODE);
      runs_.back().set_arg(1, instr_bos_.at(context_id));
      runs_.back().set_arg(2, instr_bos_.at(context_id).size() /
                                  sizeof(std::uint32_t));
      runs_.back().set_arg(3, input_bo_.address() + DDR_AIE_ADDR_OFFSET +
                                  padded_ifm_tile_offsets.at(global_tile_idx));
      runs_.back().set_arg(4, const_bo_.address() + DDR_AIE_ADDR_OFFSET);
      runs_.back().set_arg(5, output_bo_.address() + DDR_AIE_ADDR_OFFSET +
                                  global_tile_idx * padded_ofm_tile_size);
      runs_.back().set_arg(6, scratch_bos_.at(context_id).address() +
                                  DDR_AIE_ADDR_OFFSET);
#ifdef XRT_RUNLIST_EN
      runlists_.at(context_id).add(runs_.back());
#endif
      global_tile_idx++;
    }
  }

  RYZENAI_LOG_TRACE("FusionRuntime : Setup XRT Run objects ... DONE!");
}

void FusionSubgraphRuntime::copy_input(const std::vector<Tensor> &inputs) {
  constexpr std::uint32_t act_idx = 0;
  const auto act_shape = inputs.at(act_idx).shape;
  const auto act_size = std::accumulate(act_shape.begin(), act_shape.end(),
                                        size_t{1}, std::multiplies{}) *
                        Utils::get_size_of_type(inputs.at(act_idx).dtype);

  DD_THROW_IF(attr_.find("ifm_addr") == attr_.end(),
              "Can't find ifm_addr attribute for the subgraph");
  size_t offset = std::any_cast<size_t>(attr_.find("ifm_addr")->second);

  DD_THROW_IF(attr_.find("ifm_tile_shape") == attr_.end(),
              "Can't find ifm_tile_shape attribute for the subgraph");
  DD_THROW_IF(attr_.find("padded_ifm_tile_shape") == attr_.end(),
              "Can't find padded_ifm_tile_shape attribute for the subgraph");

  std::vector<size_t> ifm_tile_shape =
      std::any_cast<std::vector<size_t>>(attr_.find("ifm_tile_shape")->second);
  std::vector<size_t> padded_ifm_tile_shape =
      std::any_cast<std::vector<size_t>>(
          attr_.find("padded_ifm_tile_shape")->second);
  DD_THROW_IF(ifm_tile_shape.size() != padded_ifm_tile_shape.size(),
              "Expect num dims to be same");
  DD_THROW_IF(ifm_tile_shape.size() != 3, "Expect shape to be 3 dim");

  DD_THROW_IF(act_shape.at(3) != ifm_tile_shape.at(2), "Inconsistent C dim");

  bool need_pad = (ifm_tile_shape.at(0) != padded_ifm_tile_shape.at(0)) ||
                  (ifm_tile_shape.at(1) != padded_ifm_tile_shape.at(1)) ||
                  (ifm_tile_shape.at(2) != padded_ifm_tile_shape.at(2));

  if (need_pad) {
    RYZENAI_LOG_TRACE("FusionRuntime : copy input needs dim pad");
    DD_THROW_IF(ifm_tile_shape.at(0) > padded_ifm_tile_shape.at(0),
                "Padded dim should be larger");
    DD_THROW_IF(ifm_tile_shape.at(1) > padded_ifm_tile_shape.at(1),
                "Padded dim should be larger");
    DD_THROW_IF(ifm_tile_shape.at(2) > padded_ifm_tile_shape.at(2),
                "Padded dim should be larger");

    std::uint8_t *src = (std::uint8_t *)inputs.at(act_idx).data;
    const size_t data_transfer_size =
        act_shape.at(3) * Utils::get_size_of_type(inputs.at(act_idx).dtype);
    const size_t stride_w = padded_ifm_tile_shape.at(2) *
                            Utils::get_size_of_type(inputs.at(act_idx).dtype);
    const size_t stride_h = padded_ifm_tile_shape.at(1) * stride_w;

    for (size_t h_idx = 0; h_idx < act_shape.at(1); h_idx++) {
      const auto curr_offset = offset;
      for (size_t w_idx = 0; w_idx < act_shape.at(2); w_idx++) {
        input_bo_.write(src, data_transfer_size, offset);
        src += data_transfer_size;
        offset += stride_w;
      }
      offset = curr_offset + stride_h;
    }

    input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  } else {
    write_to_bo(input_bo_, offset, inputs.at(act_idx).data, act_size);
  }
}

void FusionSubgraphRuntime::execute_subgraph(
    const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs) {

  auto exec_start = GET_ELAPSED_TIME_NS();
  DD_THROW_IF(inputs.size() != 1, "Expect only single input subgraph");
  DD_THROW_IF(outputs.size() != 1, "Expect only single output subgraph");
  std::lock_guard<std::mutex> guard(execute_mutex_);
  // initialize inputs?
  // copy directly to BO for now.
  // also need to merge inputs.
  auto input_copy_start = GET_ELAPSED_TIME_NS();

  copy_input(inputs);

  auto input_copy_end = GET_ELAPSED_TIME_NS();

  const size_t num_sched_contexts = sched_num_tiles_per_context_.size();

  auto kernel_start = GET_ELAPSED_TIME_NS();

  try {
#ifdef XRT_RUNLIST_EN
    for (std::uint32_t context_id = 0; context_id < num_sched_contexts;
         context_id++) {
      runlists_.at(context_id).execute();
    }

    for (std::uint32_t context_id = 0; context_id < num_sched_contexts;
         context_id++) {
      runlists_.at(context_id).wait();
    }
#else
    const auto max_tiles_per_context = sched_num_tiles_per_context_.at(0);
    for (std::uint32_t tile_idx = 0; tile_idx < max_tiles_per_context;
         tile_idx++) {
      for (std::uint32_t context_id = 0; context_id < num_sched_contexts;
           context_id++) {
        if (tile_idx < sched_num_tiles_per_context_.at(context_id)) {
          runs_.at(prefix_sum_sched_num_tiles_.at(context_id) + tile_idx)
              .start();
        }
      }

      for (std::uint32_t context_id = 0; context_id < num_sched_contexts;
           context_id++) {
        if (tile_idx < sched_num_tiles_per_context_.at(context_id)) {
          runs_.at(prefix_sum_sched_num_tiles_.at(context_id) + tile_idx)
              .wait2();
        }
      }
    }
#endif
  } catch (const std::exception &e) {
#ifdef RYZENAI_DEBUG
    std::cout << "Running under debug mode...  Hardware context handle = "
              << ctxs_.at(0).get_handle() << ", PID = " << Platform::get_pid()
              << std::endl;
    std::cout << "Will wait for user input." << std::endl;
    std::cin.get();
#endif

    std::cerr << "ERROR: Kernel timed out!" << std::endl;
    std::cerr << "Details: " << e.what() << std::endl;

    xrt::error err = xrt::error(ctxs_.at(0).get_device(), XRT_ERROR_CLASS_AIE);
    if (err.get_error_code()) {
      std::string err_message =
          std::string("Error while running, info: ") + err.to_string();
      std::cerr << err_message << std::endl;
      RYZENAI_LOG_TRACE(err_message);
    }

    DD_THROW(OpsFusion::dd_format("Kernel partition timeout (Detail : {})",
                                  e.what()));
  }

  auto kernel_end = GET_ELAPSED_TIME_NS();

  // unpack outputs
  // memcpy for now
  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint8_t *out_ptr = output_bo_.map<uint8_t *>();
  // auto out_idx = 0;
  // auto out_shape = outputs.at(out_idx).shape;
  // auto out_size = std::accumulate(out_shape.begin(), out_shape.end(),
  // size_t{1},
  //                                 std::multiplies{}) *
  //                 Utils::get_size_of_type(outputs.at(out_idx).dtype);
  // std::memcpy(outputs.at(0).data, out_ptr, out_size);
  auto sg = std::make_unique<ryzenai::xcom::subgraph>(false);
  sg->format_output(outputs.at(0), out_ptr, 0, 0, attr_);

  auto output_copy_end = GET_ELAPSED_TIME_NS();

  auto exec_time = (output_copy_end - exec_start) / 1000;
  auto input_copy_time = (input_copy_end - input_copy_start) / 1000;
  auto kernel_time = (kernel_end - kernel_start) / 1000;
  auto output_copy_time = (output_copy_end - kernel_end) / 1000;

  RYZENAI_LOG_INFO("FusionSubgraphRuntime::execute_subgraph time(us)," +
                   std::to_string(exec_time) + "," +
                   std::to_string(input_copy_time) + "," +
                   std::to_string(kernel_time) + "," +
                   std::to_string(output_copy_time) + "\n");
}

void FusionSubgraphRuntime::allocate_data_bos(
    const std::vector<OpArgMap> &arg_map) {
  std::set<std::int32_t> arg_types;

  // NOTE: data BOs will be shared across xrt hw contexts
  std::uint32_t context_id = 0;

  size_t scratch_size = 0;
  bool has_scratch = false;

  scratch_bos_.clear();

  for (auto arg : arg_map) {
    DD_THROW_IF((arg_types.find((std::int32_t)arg.arg_type) != arg_types.end()),
                "Arg type shows up more than once")
    arg_types.insert((std::int32_t)arg.arg_type);
    if (arg.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      const_bo_ =
          xrt::bo(ctxs_.at(context_id), arg.size, xrt::bo::flags::host_only,
                  kernels_.at(context_id).group_id(HOST_BO_GROUP_ID));
      // std::cout << "Const BO Size: " << arg.size << std::endl;
    } else if (arg.arg_type == OpArgMap::OpArgType::INPUT) {
      input_bo_ =
          xrt::bo(ctxs_.at(context_id), arg.size, xrt::bo::flags::host_only,
                  kernels_.at(context_id).group_id(HOST_BO_GROUP_ID));
      // std::cout << "Input BO Size: " << arg.size << std::endl;
    } else if (arg.arg_type == OpArgMap::OpArgType::OUTPUT) {
      output_bo_ =
          xrt::bo(ctxs_.at(context_id), arg.size, xrt::bo::flags::host_only,
                  kernels_.at(context_id).group_id(HOST_BO_GROUP_ID));
      // std::cout << "Output BO Size: " << arg.size << std::endl;
    } else if (arg.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
      DD_THROW_IF(has_scratch, "Only expect 1 scratch buffer");
      scratch_size = std::max(arg.size, XRT_BO_MIN_SIZE);
      scratch_bos_.emplace_back(
          xrt::bo(ctxs_.at(context_id), scratch_size, xrt::bo::flags::host_only,
                  kernels_.at(context_id).group_id(HOST_BO_GROUP_ID)));

      has_scratch = true;
    }
  }

  // make sure to allocate default scratch, even if op doesnt request it
  // avoid data BO of size 0
  uint32_t scratch_start_context_id = has_scratch ? 1 : 0;
  for (context_id = scratch_start_context_id; context_id < num_contexts_;
       context_id++) {
    scratch_bos_.emplace_back(
        xrt::bo(ctxs_.at(context_id), scratch_size, xrt::bo::flags::host_only,
                kernels_.at(context_id).group_id(HOST_BO_GROUP_ID)));
  }
}
void FusionSubgraphRuntime::init_const_inputs(
    const std::vector<Tensor> &tensors,
    const std::map<std::string, std::any> &attr) {
  constexpr bool load_xrt = false;
  auto sg = std::make_unique<ryzenai::xcom::subgraph>(load_xrt);
  void *const_bo_ptr = const_bo_.map();
  auto bo_const = BoConst(const_bo_ptr);
  sg->initialize_const_params(bo_const, tensors, attr);
  const_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}
void FusionSubgraphRuntime::setup_instr_bo(std::vector<uint8_t> &txn) {
  auto ibuf_op = transaction_op(txn);

  size_t instr_bo_size = ibuf_op.get_txn_instr_size();

  std::lock_guard<std::mutex> guard(instr_state_mutex);

  std::uint32_t idx = 0;
  for (const auto &instr_bo : instr_bos_) {
    xrt_core::hwctx_handle *handle =
        static_cast<xrt_core::hwctx_handle *>(ctxs_.at(idx));
    xrt_instr_state.at(handle).heap_total_size -=
        Utils::align_to_next(instr_bo.size(), INSTR_XRT_BO_ALIGNMENT);
    xrt_instr_state.at(handle).num_instr_bos -= 1;
    idx += 1;
  }

  instr_bos_.clear();
  instr_bos_.reserve(num_contexts_);

  for (std::uint32_t context_id = 0; context_id < num_contexts_; context_id++) {
    xrt_core::hwctx_handle *handle =
        static_cast<xrt_core::hwctx_handle *>(ctxs_.at(context_id));
    // TO DO: wrap this in try catch block since we have no way of querying
    //        if this request can be satisfied
    instr_bos_.emplace_back(xrt::bo(ctxs_.at(context_id), instr_bo_size,
                                    xrt::bo::flags::cacheable,
                                    kernels_.at(context_id).group_id(1)));
    write_to_bo(instr_bos_.back(), 0 /*offset*/, ibuf_op.get_txn_op().data(),
                instr_bo_size);
    xrt_instr_state.at(handle).heap_total_size +=
        Utils::align_to_next(instr_bo_size, INSTR_XRT_BO_ALIGNMENT);
    xrt_instr_state.at(handle).num_instr_bos += 1;
  }
}

/**
 * relocate_ctrl_pkt_patch_info must be called before this api is invoked.
 */
void FusionRuntime::patch_ctrl_pkt(const Metadata &meta) {
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Control packet patch Init.."));
  uint8_t *bo_map = super_instr_bo_.map<uint8_t *>();
  uint64_t ctrl_pkt_bo_offset = MAP_AT(meta.fused_tensors, "super_instr").size;
  uint64_t input_bo_addr = input_bo_.address() + DDR_AIE_ADDR_OFFSET;
  uint64_t output_bo_addr = output_bo_.address() + DDR_AIE_ADDR_OFFSET;
  uint64_t scratch_bo_addr = scratch_bo_.address() + DDR_AIE_ADDR_OFFSET;
  uint64_t const_bo_addr = const_bo_->address() + DDR_AIE_ADDR_OFFSET;
  uint64_t super_instr_bo_addr =
      super_instr_bo_.address() + DDR_AIE_ADDR_OFFSET;

  auto patch_bd_addr = [](uint8_t *dest, uint64_t ddr_addr) {
    uint32_t addr_low = (uint32_t)(ddr_addr & 0xFFFFFFFF);
    uint16_t addr_high = (uint16_t)((ddr_addr & 0x0000FFFF00000000ULL) >> 32);
    *(uint32_t *)(dest) = addr_low;
    *(uint16_t *)(dest + 4) = addr_high;
  };

  for (auto &op : meta.op_list) {
    auto &patch_info = op.ctrl_pkt_patch_info;
    // if ctrl packet ddoes not exist for an op, skip the patching
    if ((meta.ctrl_pkt_map.find(op.name) == meta.ctrl_pkt_map.end()) ||
        (!patch_info.size())) {
      continue;
    }
    auto op_offset = meta.ctrl_pkt_map.at(op.name).offset + ctrl_pkt_bo_offset;
    RYZENAI_LOG_TRACE(OpsFusion::dd_format(
        "Final DDR address patch: op_name: {}, op_type: {}", op.name, op.type));

    for (auto &patch : patch_info) {
      auto offset = op_offset + patch.offset;
      uint64_t *ptr = (uint64_t *)(bo_map + offset);
      switch (patch.xrt_arg_idx) {
      case OpArgMap::INPUT: {
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching input arg. arg_idx: {}, bo_offset: {}, input_bo_addr: "
            "{}, at [offset, addr]: [{}, {}], patch.offset: {}",
            patch.xrt_arg_idx, patch.bo_offset, input_bo_addr, offset, ptr,
            patch.offset));
        patch_bd_addr((uint8_t *)ptr, patch.bo_offset + input_bo_addr);
        break;
      }
      case OpArgMap::CONST_INPUT: {
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching const arg. arg_idx: {}, bo_offset: {}, const_bo_addr: "
            "{}, at [offset, addr]: [{}, {}], patch.offset: {}",
            patch.xrt_arg_idx, patch.bo_offset, const_bo_addr, offset, ptr,
            patch.offset));
        patch_bd_addr((uint8_t *)ptr, patch.bo_offset + const_bo_addr);
        break;
      }
      case OpArgMap::CONST_KERNEL_PARAM_INPUT: {
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching super_instr arg. arg_idx: {}, bo_offset: {}, "
            "super_instr_bo_addr: {}, at [offset, addr]: [{}, {}], "
            "patch.offset: {}",
            patch.xrt_arg_idx, patch.bo_offset, super_instr_bo_addr, offset,
            ptr, patch.offset));
        patch_bd_addr((uint8_t *)ptr, patch.bo_offset + super_instr_bo_addr);
        break;
      }
      case OpArgMap::CTRL_PKT_BIN: {
        RYZENAI_LOG_TRACE(
            OpsFusion::dd_format("Patching control_pkt arg. arg_idx: {}, "
                                 "bo_offset: {}, super_instr_bo_addr: {}, at "
                                 "[offset, addr]: [{}, {}], patch.offset: {}",
                                 patch.xrt_arg_idx, patch.bo_offset,
                                 super_instr_bo_addr + ctrl_pkt_bo_offset,
                                 offset, ptr, patch.offset));
        patch_bd_addr((uint8_t *)ptr, patch.bo_offset + super_instr_bo_addr +
                                          ctrl_pkt_bo_offset);
        break;
      }
      case OpArgMap::SCRATCH_PAD: {
        RYZENAI_LOG_TRACE(
            OpsFusion::dd_format("Patching scratch arg. arg_idx: {}, "
                                 "bo_offset: {}, scratch_bo_addr: {}, at "
                                 "[offset, addr]: [{}, {}], patch.offset: {}",
                                 patch.xrt_arg_idx, patch.bo_offset,
                                 scratch_bo_addr, offset, ptr, patch.offset));
        patch_bd_addr((uint8_t *)ptr, patch.bo_offset + scratch_bo_addr);
        break;
      }
      case OpArgMap::OUTPUT: {
        RYZENAI_LOG_TRACE(OpsFusion::dd_format(
            "Patching output arg. arg_idx: {}, bo_offset: {}, output_bo_addr: "
            "{}, at [offset, addr]: [{}, {}], patch.offset: {}",
            patch.xrt_arg_idx, patch.bo_offset, output_bo_addr, offset, ptr,
            patch.offset));
        patch_bd_addr((uint8_t *)ptr, patch.bo_offset + output_bo_addr);
        break;
      }
      default:
        DD_THROW(dd_format("Unknown arg type for op {}", op.name));
        break;
      }
    }
  }

  super_instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Control packet patch Done.."));
}

} // namespace OpsFusion
