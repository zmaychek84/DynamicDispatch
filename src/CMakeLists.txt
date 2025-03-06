# Copyright (c) 2025 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

set(LIBRARY_NAME dyn_dispatch_core)

#find_package(AIEBU REQUIRED)

function(dd_add_protobuf_target target proto_file)
  if(COMMAND protobuf_generate)
    add_library(${target} OBJECT ${proto_file})
    protobuf_generate(
      TARGET
      ${target}
      LANGUAGE
      cpp
      PROTOC_OPTIONS
      "$<LIST:TRANSFORM,$<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
    )
  else()
    get_filename_component(PROTO_FILE_NAME ${proto_file} NAME_WE)
    set(PROTO_SRCS
        ${CMAKE_CURRENT_BINARY_DIR}/fusion_rt/${PROTO_FILE_NAME}.pb.cc
    )
    set(PROTO_HDRS
        ${CMAKE_CURRENT_BINARY_DIR}/fusion_rt/${PROTO_FILE_NAME}.pb.h
    )
    add_custom_command(
      OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND
        protobuf::protoc ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I
        ${protobuf_SOURCE_DIR}/src -I ${CMAKE_CURRENT_SOURCE_DIR}/
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${proto_file}
      DEPENDS ${proto_file}
    )
    add_library(${target} OBJECT ${PROTO_SRCS} ${PROTO_HDRS})
  endif()
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_include_directories(
    ${target} SYSTEM
    PRIVATE
      $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_link_libraries(${target} INTERFACE protobuf::libprotobuf ${ABSL_LIB})
endfunction()

dd_add_protobuf_target(dd_metastate_proto fusion_rt/metastate.proto)
if(NOT WIN32)
  add_compile_options(-Wno-unused-result)
endif()

set(LIB_SRC
    ops/op_interface.cpp
    ops/op_builder.cpp
    ops/matmul/matmul.cpp
    ops/dmacompiler/batch_matmul/batch_matmul.cpp
    ops/maxpool/maxpool.cpp
    ops/act_act_matmul_qdq/act_act_matmul_qdq.cpp
    ops/act_const_add/act_const_add.cpp
    ops/act_matmul_softmax/act_matmul_softmax.cpp
    ops/layernorm/layernorm.cpp
    ops/groupnorm/groupnorm.cpp
    ops/mhagprb/mhagprb.cpp
    ops/mhachannel/mhachannel.cpp
    ops/mhawindow/mhawindow.cpp
    ops/mha/mha.cpp
    ops/dmacompiler/mhapsw/mhapsw.cpp
    ops/dmacompiler/qdq_mul/qdq_mul.cpp
    ops/mhamzdk5/mhamzdk5.cpp
    ops/matmulgeluadd/matmulgeluadd.cpp
    ops/dmacompiler/matmul_v_geluadd/matmul_v_geluadd.cpp
    ops/matmulbias/matmulbias.cpp
    ops/matmul_a16a16_mladf/matmul_a16a16_mladf.cpp
    ops/matmul_a16w8_mladf/matmul_a16w8_mladf.cpp
    ops/elwadd/elwadd.cpp
    ops/dmacompiler/qdqadd/qdqadd.cpp
    ops/dmacompiler/gather_qdq_add/gather_qdq_add.cpp
    ops/mladfadd/mladfadd.cpp
    ops/transpose/transpose.cpp
    ops/silu/silu.cpp
    ops/gelu_e/gelue.cpp
    ops/unary/unary.cpp
    ops/elwmul/elwmul.cpp
    ops/elwdiv_qdq/elwdiv_qdq.cpp
    ops/maskedsoftmax/maskedsoftmax.cpp
    ops/mladfmharope/mladfmharope.cpp
    ops/mladfrmsnorm/mladfrmsnorm.cpp
    ops/softmax_qdq/softmax_qdq.cpp
    ops/experimental/cube.cpp
    ops/experimental/square.cpp
    ops/experimental/matmul_tile.cpp
    ops/experimental/elwadd_tile.cpp
    ops/pm_load/pm_load.cpp
    ops/record_timer/record_timer.cpp
    ops/preemption/preemption.cpp
    ops/mladfmatmulbias/mladfmatmulbias.cpp
    ops/bmm/bmm.cpp
    ops/mladfsoftmax/mladfsoftmax.cpp
    ops/mladfelwadd/mladfelwadd.cpp
    ops/mladfelwmul/mladfelwmul.cpp
    fusion_rt/fusion_rt.cpp
    fusion_rt/meta_utils.cpp
    fusion_rt/metastate_api.cpp
    fusion_rt/md5.cpp
    fusion_rt/md5.h
    fusion_rt/weak.hpp
    passes/insert_pm_swap.cpp
    passes/remove_identity_ops.cpp
    passes/insert_record_timer.cpp
    passes/insert_preemption_pass.cpp
    passes/assign_pdi_id_pass.cpp
    passes/generate_pdi_partitions_pass.cpp
    passes/analyze_buffer_reqs.cpp
    passes/optimize_scratch.cpp
    passes/split_max_partition_pass.cpp
    passes/sync_identity_tensors.cpp
    passes/fetch_op_txn_bins.cpp
    passes/relocate_ctrl_pkt_patch_info.cpp
    passes/cont_reuse.hpp
    passes/cont_reuse.cpp
    passes/optimize_scratch_bucket.cpp
    passes/optimize_scratch_contiguous.cpp
    txn/txn_utils.cpp
    utils/logging.cpp
    utils/xrt_context.cpp
    utils/instruction_registry.cpp
    utils/instruction_cache.cpp
    utils/op_utils.cpp
    utils/ctrl_pkt_utils.cpp
    utils/platform.cpp
    utils/pass_utils.cpp
    ops/conv/conv.cpp
    ops/convForMatmulAdd/convForMatmulAdd.cpp
    ops/concateOps/concateOps.cpp
    ops/gap/gap.cpp
    ops/iconv/iconv.cpp
    ops/conv2matmul/conv2matmul.cpp
    ops/conv2matmul_silu/conv2matmul_silu.cpp
    ops/gelu/gelu.cpp
    ops/silu_qdq/silu_qdq.cpp
    ops/expand/expand.cpp
    ops/sigmoid/sigmoid.cpp
    ops/l2_norm/l2_norm.cpp
    ops/DotProductSigmoid/DotProductSigmoid.cpp
    ops/ReduceSum/ReduceSum.cpp
    ops/AddTanhLPNorm/AddTanhLPNorm.cpp
    ops/AttentionMaskPrePro/AttentionMaskPrePro.cpp
    ops/dmacompiler/AttentionMaskPrePro_win25/AttentionMaskPrePro_win25.cpp
    ops/matvecadd/matvecadd.cpp
    ops/elwmul_qdq/elwmul_qdq.cpp
    ops/nni_resize/nni_resize.cpp
    ops/concat/concat.cpp
    ops/slice/slice.cpp
    ops/xcom/conv/weight_shuffle.cpp
    ops/xcom/conv/conv.cpp
    ops/xcom/subgraph/subgraph.cpp
    ops/quant/quant.cpp
    ops/dequant/dequant.cpp
    txn_helper/txn_helper.cpp
    txn_helper/txn_tiling_util.cpp
    utils/utils.cpp
    utils/tmpfile.cpp
    ops/lstm/lstm.cpp
    $<TARGET_OBJECTS:dd_metastate_proto>
    ops/conv/conv_lp.cpp
    ops/gap/gap_lp.cpp
    ops/sd/conv2d.cpp
    ops/sd/concat.cpp
    ops/sd/matmul.cpp
    ops/sd/gelu.cpp
    ops/sd/silu.cpp
    ops/sd/elwadd.cpp
    ops/sd/elwmul.cpp
    ops/sd/resize.cpp
    ops/sd/layernorm.cpp
    ops/sd/mha.cpp
    ops/ops_common/coeffs.cpp
    ops/sd/slice.cpp
    ops/sd/groupnorm.cpp
    ops/sd/gemm.cpp
    ops/flat/mlp.cpp
    ops/flat/mha_v2.cpp
    ops/flat/rms_add.cpp
    ops/cast/cast.cpp
    ops/matmul_cpu/matmul_cpu.cpp
    ops/quantizelinear_cpu/quantizelinear_cpu.cpp
    ops/dequantizelinear_cpu/dequantizelinear_cpu.cpp
    ops/identity/identity.cpp
)

# Hook for simnow lite
set(HWEMU "")
if(ENABLE_SIMNOWLITE_BUILD)
  set(HWEMU ${XRT_HWEMU_LIBRARIES})
endif()
add_library(${LIBRARY_NAME} ${LIB_SRC})

if(MSVC)
  target_compile_options(${LIBRARY_NAME} PRIVATE /MP)
endif()

set(AIEBU_CORE_LIB "")
if(WIN32 AND (NOT DD_DISABLE_AIEBU))
  set(AIEBU_CORE_LIB AIEBU::aiebu_static)
else()
  set(AIEBU_INCLUDE_DIRS "")
endif()

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PUBLIC $<INSTALL_INTERFACE:include/ryzenai/dynamic_dispatch>
  PUBLIC $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE ${XRT_INCLUDE_DIRS}
  PRIVATE ${DD_SRC_INCLUDE_DIRS}
  PRIVATE ${AIEBU_INCLUDE_DIRS}
)

target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC nlohmann_json::nlohmann_json
         spdlog::spdlog
         xaiengine::xaiengine
         transaction
         xclbin
         pm_package
         ${AIEBU_CORE_LIB}
         $<TARGET_PROPERTY:dd_metastate_proto,INTERFACE_LINK_LIBRARIES>
  PRIVATE ${XRT_COREUTIL_LIBRARIES} ${HWEMU}
  # $<TARGET_PROPERTY:dd_metastate_proto,LINK_LIBRARIES>
)

target_compile_options(${LIBRARY_NAME} PRIVATE ${DD_DEFAULT_COMPILE_OPTIONS})
target_compile_definitions(${LIBRARY_NAME} PUBLIC XAIE_FEATURE_MSVC)
set_source_files_properties(
  fusion_rt/fusion_rt.cpp fusion_rt/metastate_api.cpp
  PROPERTIES # COMPILE_OPTIONS $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-error>
             COMPILE_OPTIONS $<$<CXX_COMPILER_ID:MSVC>:/w>
)
set_property(TARGET ${LIBRARY_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(
    ${LIBRARY_NAME} PUBLIC DYNAMIC_DISPATCH_BUILD_SHARED
  )
  target_compile_definitions(${LIBRARY_NAME} PRIVATE DYNAMIC_DISPATCH_EXPORT)
endif()

if(ENABLE_DD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
  )
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_DynamicDispatch "./python/bindings.cpp")
  target_include_directories(
    _DynamicDispatch PRIVATE ${XRT_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src
  )
  target_link_libraries(_DynamicDispatch PRIVATE ${LIBRARY_NAME})
  target_compile_options(_DynamicDispatch PRIVATE ${DD_DEFAULT_COMPILE_OPTIONS})
endif()

target_include_directories(
  ${LIBRARY_NAME} PRIVATE ${nlohmann_json_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(${LIBRARY_NAME} PRIVATE "/wd4996")
  target_compile_options(${LIBRARY_NAME} PRIVATE "/wd4703")
endif(MSVC)

include(GNUInstallDirs)
# Install libraries
if(CMAKE_SOURCE_DIR STREQUAL DynamicDispatch_SOURCE_DIR)
  install(
    TARGETS ${LIBRARY_NAME} dd_metastate_proto
    EXPORT ${PROJECT_NAME}-targets
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryzenai/dynamic_dispatch
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(ENABLE_DD_PYTHON AND SKBUILD AND CMAKE_SOURCE_DIR STREQUAL
                                    DynamicDispatch_SOURCE_DIR
)
  install(TARGETS _DynamicDispatch LIBRARY DESTINATION ".")
endif()

# Linux Can Be Fixed Later
if(MSVC AND CMAKE_SOURCE_DIR STREQUAL DynamicDispatch_SOURCE_DIR)
  install(
    EXPORT ${PROJECT_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    COMPONENT runtime
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # Support Find Package
  include(CMakePackageConfigHelpers)

  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/config.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake @ONLY
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    COMPONENT runtime DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
endif()

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryzenai/dynamic_dispatch
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
