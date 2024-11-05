# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

add_executable(test_mladfadd_tile test_fusion.cpp)
dd_configure_test(test_mladfadd_tile OFF)
