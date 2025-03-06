// Copyright (c) 2025 Advanced Micro Devices, Inc
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

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <memory>
#if defined(RYZENAI_PERF) || defined(RYZENAI_TRACE)
#define RYZENAI_LOGGING
#endif

#ifdef RYZENAI_LOGGING

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

namespace ryzenai {

constexpr auto perf_logger_name = "ryzenai_perf_logger";
constexpr auto perf_logger_fname = "logs/ryzenai_ops.log";
constexpr auto trace_logger_name = "ryzenai_trace_logger";
constexpr auto trace_logger_fname = "logs/ryzenai_trace.log";

class logger {
public:
  enum log_levels {
    PERF,
    TRACE,
  };

  static logger &get_instance();

  spdlog::logger *get(log_levels lvl) const;

  double get_elapsed();

private:
  std::shared_ptr<spdlog::logger> perf_logger_;
  std::shared_ptr<spdlog::logger> trace_logger_;
  spdlog::stopwatch sw;
  bool enable_perf_logging;
  bool enable_trace_logging;
  logger();
};
} /* namespace ryzenai */

#ifdef RYZENAI_PERF
#define GET_ELAPSED_TIME_NS() (ryzenai::logger::get_instance().get_elapsed())
#define RYZENAI_LOG_INFO(message)                                              \
  {                                                                            \
    auto *logger = ryzenai::logger::get_instance().get(                        \
        ryzenai::logger::log_levels::PERF);                                    \
    if (logger->should_log(spdlog::level::info)) {                             \
      SPDLOG_LOGGER_INFO(logger, message);                                     \
    }                                                                          \
  }
#endif /* RYZENAI_PERF */

#ifdef RYZENAI_TRACE
#define RYZENAI_LOG_TRACE(message)                                             \
  {                                                                            \
    auto *logger = ryzenai::logger::get_instance().get(                        \
        ryzenai::logger::log_levels::TRACE);                                   \
    if (logger->should_log(spdlog::level::trace)) {                            \
      SPDLOG_LOGGER_TRACE(logger, message);                                    \
    }                                                                          \
  }
#endif /*RYZENAI_TRACE */

#endif /* RYZENAI_LOGGING */

#ifndef RYZENAI_PERF
#define GET_ELAPSED_TIME_NS() 0
#define RYZENAI_LOG_INFO(message)
#endif

#ifndef RYZENAI_TRACE
#define RYZENAI_LOG_TRACE(message)
#endif

#endif /* __LOGGING_HPP__ */
