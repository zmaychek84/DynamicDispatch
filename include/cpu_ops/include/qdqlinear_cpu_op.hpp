/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#ifndef __CPU_OP_QDQLINEAR_
#define __CPU_OP_QDQLINEAR_

#pragma once

//#include "vaip/vaip.hpp"
//#include "vart/runner_ext.hpp"
#include <algorithm>
#include <chrono>
#include <future>
#include <mutex>

namespace cpu_runner_ops {
// #define _QDQ_MT_ 0
#define MT_POINT 800000

// Convert float to T with rounding
template <typename T> static inline T rounder(float data) {
  static const int data_max = std::numeric_limits<T>::max();
  static const int data_min = std::numeric_limits<T>::min();
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if ((data - floor(data)) == 0.5) {
    rlt = std::round(data * 0.5f) * 2.0f;
  } else {
    rlt = static_cast<T>(round(data));
  }
  return rlt;
}

template <typename OutputType>
inline void qlinear_op(const float* src, OutputType* dst,
                       std::size_t num_elements, const float scale,
                       const int zero_point) {

  static const int maxVal = std::numeric_limits<OutputType>::max();
  static const int minVal = std::numeric_limits<OutputType>::min();

  std::size_t remainder = num_elements;

#ifdef _WIN32
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);

  // to check
  // cout << string(typeid(OutputType).name()) << endl;

  __m512i min_vec = _mm512_set1_epi32(minVal);
  __m512i max_vec = _mm512_set1_epi32(maxVal);

  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    in = _mm512_roundscale_ps(_mm512_div_ps(in, scale_vector),
                              _MM_FROUND_TO_NEAREST_INT);
    __m512i in32 = _mm512_add_epi32(_mm512_cvtps_epi32(in), zero_point_vector);
    __m512i clamped =
        _mm512_min_epi32(_mm512_max_epi32(in32, min_vec), max_vec);

    if constexpr (sizeof(OutputType) == 1) {        // int8 / uint8
      _mm_storeu_epi8(dst, _mm512_cvtepi32_epi8(clamped));
    } else if constexpr (sizeof(OutputType) == 2) { // int16 / uint16
      _mm256_storeu_epi16(dst, _mm512_cvtepi32_epi16(clamped));
    }

    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }
#endif
  
  if (remainder > 0) {
    // std::transform(src, src + remainder, dst, [&](const float& src) { return
    // rounder<OutputType>((src / scale) + (float)zero_point); });
    std::transform(src, src + remainder, dst, [&](const float& src) {
      float FloatValue;
      FloatValue = std::nearbyintf(src / scale) + (float)zero_point;
      FloatValue = std::max(FloatValue, (float)minVal);
      FloatValue = std::min(FloatValue, (float)maxVal);
      return (OutputType)(int32_t)FloatValue;
    });
  }
}

template <typename InType>
void dqlinear_op(const InType* src, float* dst, std::size_t num_elements,
                 float scale, int zero_point) {
  
  std::size_t remainder = num_elements;
  
#ifdef _WIN32
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m512i zero_point_vector = _mm512_set1_epi32((int)zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    __m512i in32;
    if constexpr (std::is_same_v<InType, signed char>) {
      _mm_prefetch((const char*)src + 16, _MM_HINT_T0);
      in32 = _mm512_cvtepi8_epi32(_mm_loadu_epi8(src));
    } else if constexpr (std::is_same_v<InType, unsigned char>) {
      _mm_prefetch((const char*)src + 16, _MM_HINT_T0);
      in32 = _mm512_cvtepu8_epi32(_mm_loadu_epi8(src));
    }
    if constexpr (std::is_same_v<InType, short>) {
      _mm_prefetch((const char*)src + 32, _MM_HINT_T0);
      in32 = _mm512_cvtepi16_epi32(_mm256_loadu_epi16(src));
    } else if constexpr (std::is_same_v<InType, unsigned short>) {
      _mm_prefetch((const char*)src + 32, _MM_HINT_T0);
      in32 = _mm512_cvtepu16_epi32(_mm256_loadu_epi16(src));
    }

    __m512 mul = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_sub_epi32(in32, zero_point_vector)),
        scale_vector);
    _mm512_storeu_ps(dst, mul);
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }
#endif

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const InType& src) {
      return (((int)src - zero_point) * scale);
    });
  }
}

template <typename OutputType>
inline void QuantizeLinear(const float* Input, OutputType* Output, size_t N,
                           float Scale, int ZeroPoint, std::vector<std::future<int>>& thr_fut = *(new std::vector<std::future<int>>)) {
// #if _QDQ_MT_
  if ((N > MT_POINT) && (thr_fut.size() > 0)) {
    auto THREAD_NUM = thr_fut.size();
    size_t THREAD_WORKLOAD = size_t(ceil((float)N / THREAD_NUM));
    for (size_t i = 0U; i < THREAD_NUM; i++) {
      thr_fut[i] = std::async(
          std::launch::async,
          [&](size_t i) {
            size_t BASE_POS = i * THREAD_WORKLOAD;
            auto workload = std::min(THREAD_WORKLOAD + BASE_POS, N);
            /*float FloatValue;
            for (size_t n = BASE_POS; n < workload; n++) {
              FloatValue = std::nearbyintf(Input[n] / Scale) + ZeroPoint;
              FloatValue = std::max(FloatValue, MinimumValue);
              FloatValue = std::min(FloatValue, MaximumValue);
              Output[n] = (OutputType)(int32_t)FloatValue;
            }*/

            if (workload > BASE_POS)
              qlinear_op((Input + BASE_POS), (Output + BASE_POS),
                        (workload - BASE_POS), Scale, (int)ZeroPoint);

            return 0;
          },
          i);
    }

    for (auto i = 0U; i < THREAD_NUM; i++) {
      thr_fut[i].wait();
    }
  } else {
  // #else
    qlinear_op(Input, Output, N, Scale, (int)ZeroPoint);
  // #endif
  }
}

template <typename InType>
inline void DequantizeLinear(const InType* Input, float* Output, std::size_t N,
                             float Scale, int ZeroPoint, std::vector<std::future<int>>& thr_fut = *(new std::vector<std::future<int>>)) {
// #if _QDQ_MT_
  if ((N > MT_POINT) && (thr_fut.size() > 0)) {
    auto THREAD_NUM = thr_fut.size();
    size_t THREAD_WORKLOAD = size_t(ceil((float)N / THREAD_NUM));
    for (size_t i = 0U; i < THREAD_NUM; i++) {
      thr_fut[i] = std::async(
          std::launch::async,
          [&](size_t i) {
            size_t BASE_POS = i * THREAD_WORKLOAD;
            auto workload = std::min(THREAD_WORKLOAD + BASE_POS, N);
            if (workload > BASE_POS)
              dqlinear_op((Input + BASE_POS), (Output + BASE_POS),
                          (workload - BASE_POS), Scale, (int)ZeroPoint);

            return 0;
          },
          i);
    }

    for (auto i = 0U; i < THREAD_NUM; i++) {
      thr_fut[i].wait();
    }
  // #else
  } else {
    dqlinear_op(Input, Output, N, Scale, (int)ZeroPoint);
  // #endif
  }
}
} // end of namespace

#endif //__CPU_OP_QDQLINEAR_
