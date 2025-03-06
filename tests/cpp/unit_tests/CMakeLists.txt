# Copyright (c) 2025 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_executable(
  cpp_tests
  ddsw/test_cont_reuse.cpp
  experimental/test_cube.cpp
  experimental/test_elwadd_tile.cpp
  experimental/test_matmul_tile.cpp
  experimental/test_mladfmatmulbias_tile.cpp
  experimental/test_square.cpp
  test_act_act_matmul.cpp
  test_act_const_add.cpp
  test_act_matmul_softmax.cpp
  test_AddTanhLPNorm.cpp
  test_AttentionMaskPrePro.cpp
  test_AttentionMaskPrePro_win25.cpp
  test_batch_matmul.cpp
  test_bmm.cpp
  test_bmmB.cpp
  test_bmmC.cpp
  test_bmmD.cpp
  test_cast.cpp
  test_concat.cpp
  test_concateOps.cpp
  test_conv.cpp
  test_conv2matmul.cpp
  test_conv2matmul_silu.cpp
  test_convForMatmulAdd.cpp
  test_dequant.cpp
  test_DotProductSigmoid.cpp
  test_elwadd.cpp
  test_elwdiv_qdq.cpp
  test_elwmul.cpp
  test_elwmul_qdq.cpp
  test_expand.cpp
  test_flat_mha_v2.cpp
  test_flat_mlp.cpp
  test_flat_rms_add.cpp
  test_gap.cpp
  test_gatherqdqadd.cpp
  test_gelu.cpp
  test_graphMode.cpp
  test_groupnorm.cpp
  test_iconv.cpp
  test_is_supported.cpp
  test_l2_norm.cpp
  test_layernorm.cpp
  test_lstm.cpp
  test_maskedsoftmax.cpp
  test_matmul.cpp
  test_matmul_a16a16_mladf.cpp
  test_matmul_a16w8_mladf.cpp
  test_matmul_v_geluadd.cpp
  test_matmulbias.cpp
  test_matmulgeluadd.cpp
  test_matvecadd.cpp
  test_mha.cpp
  test_mhachannel.cpp
  test_mhagprb.cpp
  test_mhamzdk5.cpp
  test_mhapsw.cpp
  test_mhawindow.cpp
  test_mladfadd.cpp
  test_mladfelwadd.cpp
  test_mladfelwmul.cpp
  test_mladfmatmulbias.cpp
  test_mladfmharope.cpp
  test_mladfrmsnorm.cpp
  test_mladfsoftmax.cpp
  test_nni_resize.cpp
  test_qdq_mul.cpp
  test_qdqadd.cpp
  test_quant.cpp
  test_ReduceSum.cpp
  test_sd_concat.cpp
  test_sd_conv.cpp
  test_sd_elwadd.cpp
  test_sd_elwmul.cpp
  test_sd_gelu.cpp
  test_sd_gemm.cpp
  test_sd_groupnorm.cpp
  test_sd_layernorm.cpp
  test_sd_matmul.cpp
  test_sd_mha.cpp
  test_sd_resize.cpp
  test_sd_silu.cpp
  test_sd_slice.cpp
  test_sigmoid.cpp
  test_silu.cpp
  test_silu_qdq.cpp
  test_slice.cpp
  test_softmax_qdq.cpp
  test_transpose.cpp
  test_xcom_conv2d.cpp
)

if(MSVC)
  target_compile_options(cpp_tests PRIVATE /MP)
endif()

dd_configure_test(cpp_tests ON)
if(UNIT_TEST_PERF_EN)
  message(STATUS "Enabling performance checks in unit tests")
  target_compile_definitions(cpp_tests PRIVATE UNIT_TEST_PERF)
endif()
