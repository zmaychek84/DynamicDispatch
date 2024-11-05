/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <utils/platform.hpp>

namespace OpsFusion {
namespace Platform {

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cstddef>
#include <unistd.h>
#endif

size_t get_pid() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

} // namespace Platform
} // namespace OpsFusion
