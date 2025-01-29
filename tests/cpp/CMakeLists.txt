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

# Subdirectories
add_subdirectory(unit_tests)
add_subdirectory(metastate_fwd_compatibility)
add_subdirectory(single_matmul)
add_subdirectory(parallel_matmul)
add_subdirectory(parallel_lrn_conv)
add_subdirectory(single_add)
add_subdirectory(matmul6)
add_subdirectory(single_lrn)
add_subdirectory(l2_norm)
add_subdirectory(single_matmuladdgelu)
add_subdirectory(single_mladfmatmul_a16a16)
add_subdirectory(single_mladfmatmul_a16w8)
add_subdirectory(single_mladfmatmulbias)
add_subdirectory(single_mladfmatmulbias_tile)
add_subdirectory(parallel_mladfmatmulbias)
add_subdirectory(single_mladfmharope)
add_subdirectory(single_mladfrmsnorm)
add_subdirectory(single_mladfrmsnorm_tile)
add_subdirectory(single_mladfelwadd)
add_subdirectory(single_mladfelwmul)
add_subdirectory(single_mha)
add_subdirectory(single_bmm1)
add_subdirectory(single_bmm2)
add_subdirectory(elwmul)
add_subdirectory(elwmul_tile)
add_subdirectory(single_mladfadd)
add_subdirectory(single_mladfadd_tile)
add_subdirectory(pm_swap)
add_subdirectory(silu)
add_subdirectory(silu_tile)
add_subdirectory(xcom_conv2d)
add_subdirectory(maskedsoftmax)
add_subdirectory(single_mladfsoftmax)
add_subdirectory(multi_thread_matmul)
add_subdirectory(mlp)
add_subdirectory(mmulsilu)
add_subdirectory(mlp4of5)
add_subdirectory(mmmm)
add_subdirectory(multi_xrt_contexts)
add_subdirectory(xcom_subgraph)
add_subdirectory(gelu)
add_subdirectory(sd_conv)
add_subdirectory(sd_concat)
add_subdirectory(sd_matmul)
add_subdirectory(sd_gelu)
add_subdirectory(sd_silu)
add_subdirectory(sd_elwadd)
add_subdirectory(sd_elwmul)
add_subdirectory(sd_resize)
add_subdirectory(sd_layernorm)
add_subdirectory(sd_mha)
add_subdirectory(sd_slice)
add_subdirectory(sd_groupnorm)
add_subdirectory(sd_gemm)
add_subdirectory(flat_mha_v2)
add_subdirectory(flat_mlp)
add_subdirectory(flat_rms_add)
