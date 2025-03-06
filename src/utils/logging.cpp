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

#include "utils/logging.hpp"
#ifdef RYZENAI_LOGGING

#include <chrono>
#include <stdexcept>

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <utils/utils.hpp>

namespace ryzenai {

logger &logger::get_instance() {
  static logger logger_;
  return logger_;
}

spdlog::logger *logger::get(log_levels lvl) const {
  if (lvl == PERF) {
    return perf_logger_.get();
  } else if (lvl == TRACE) {
    return trace_logger_.get();
  } else {
    throw std::runtime_error("Invalid logger option");
  }
}

double logger::get_elapsed() {
  return sw.elapsed() / std::chrono::nanoseconds(1);
}

logger::logger() {
  std::string log_level = (Utils::get_env_var("DD_LOG_LEVEL"));
  enable_perf_logging = (log_level == "PERF") || (log_level == "ALL");
  enable_trace_logging = (log_level == "TRACE") || (log_level == "ALL");

  // dummy loggers if logging is not enabled
  perf_logger_ = spdlog::stdout_color_mt("dummy_perf");
  perf_logger_->set_level(spdlog::level::off);
  trace_logger_ = spdlog::stdout_color_mt("dummy_trace");
  trace_logger_->set_level(spdlog::level::off);

  if (enable_perf_logging) {
    perf_logger_ = spdlog::create<spdlog::sinks::basic_file_sink_mt>(
        perf_logger_name, perf_logger_fname, true);
    perf_logger_->set_formatter(
        std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter(
            "%v", spdlog::pattern_time_type::local, "")));
  }

  if (enable_trace_logging) {
    std::string to_stdout = (Utils::get_env_var("RYZENAI_LOG_stdout"));
    if (to_stdout == "1") {
      trace_logger_ = spdlog::stdout_color_mt("RYZENAI_TRACE");
    } else {
      trace_logger_ = spdlog::create<spdlog::sinks::basic_file_sink_mt>(
          trace_logger_name, trace_logger_fname, true);
    }
    trace_logger_->set_level(spdlog::level::trace);
    trace_logger_->flush_on(spdlog::level::trace);
  }
}

} // namespace ryzenai

#endif /* RYZENAI_LOGGING */
