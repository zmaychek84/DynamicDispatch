/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

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
  SPDLOG_LOGGER_INFO(                                                          \
      ryzenai::logger::get_instance().get(ryzenai::logger::log_levels::PERF),  \
      message)
#endif /* RYZENAI_PERF */

#ifdef RYZENAI_TRACE
#define RYZENAI_LOG_TRACE(message)                                             \
  SPDLOG_LOGGER_TRACE(                                                         \
      ryzenai::logger::get_instance().get(ryzenai::logger::log_levels::TRACE), \
      message)
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
