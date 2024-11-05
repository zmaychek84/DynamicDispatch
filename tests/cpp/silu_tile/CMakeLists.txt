# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(test_silu_tile test_fusion.cpp)
dd_configure_test(test_silu_tile OFF)
