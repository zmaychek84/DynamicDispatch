# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(test_matmul_cpu test_fusion.cpp)
add_executable(test_cpu_runner test_cpu_runner.cpp)
dd_configure_test(test_matmul_cpu OFF)
dd_configure_test(test_cpu_runner OFF)
