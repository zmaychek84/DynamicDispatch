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

// clang-format off

// group the strings for each option
struct command_short_help {
  const char* command_string;
  const char* short_string;
  const char* help_string;
  const char* simple_string;

  void print(std::ostream& ostr, int c1, int c2, int c3, int c4) const {
    ostr
      << std::left << std::setw(c1) << command_string       << " = "
      << std::left << std::setw(c2) << help_string          << " ( "
      << std::left << std::setw(c3) << short_string         << " ) ( "
      << std::left << std::setw(c4) << simple_string        << ") ";
  }
};

static const char* string_dd_metastate       = "dd_metastate"       ;
static const char* string_avoid              = "avoid"              ;

static const char* string_load_only          = "--load_only"        ;
static const char* string_save_only          = "--save_only"        ;
static const char* string_write_input        = "--write_input"      ;
static const char* string_write_output       = "--write_output"     ;
static const char* string_gen_golden         = "--gen_golden"       ;
static const char* string_gen_state          = "--gen_state"        ;
static const char* string_use_state          = "--use_state"        ;
static const char* string_compare_output     = "--compare"          ;
static const char* string_no_execute         = "--no_execute"       ;
static const char* string_no_avoid           = "--no_avoid"         ;
static const char* string_local_xclbin       = "--local_xclbin"     ;
static const char* string_test_configs       = "--test_all_configs" ;
static const char* string_test_config        = "--test_config"      ;
static const char* string_init_method        = "--init_method"      ;
static const char* string_print_summary      = "--print_summary"    ;
static const char* string_print_perf         = "--print_perf"       ;
static const char* string_print_debug        = "--print_debug"      ;
static const char* string_quiet              = "--quiet"            ;
static const char* string_cleanup            = "--cleanup"          ;

static const char* short_string_load_only      = "-lo"  ;
static const char* short_string_save_only      = "-so"  ;
static const char* short_string_write_input    = "-wi"  ;
static const char* short_string_write_output   = "-wo"  ;
static const char* short_string_gen_golden     = "-gg"  ;
static const char* short_string_gen_state      = "-gs"  ;
static const char* short_string_use_state      = "-us"  ;
static const char* short_string_compare_output = "-co"  ;
static const char* short_string_no_execute     = "-nox" ;
static const char* short_string_no_avoid       = "-all" ;
static const char* short_string_local_xclbin   = "-lx"  ;
static const char* short_string_test_configs   = "-tcs" ;
static const char* short_string_test_config    = "-tc"  ;
static const char* short_string_init_method    = "-im"  ;
static const char* short_string_print_summary  = "-ps"  ;
static const char* short_string_print_perf     = "-pp"  ;
static const char* short_string_print_debug    = "-pd"  ;
static const char* short_string_quiet          = "-q"   ;
static const char* short_string_cleanup        = "-cleanup"  ;

static const char* help_string_load_only           = "do not save metastate files"                     ;
static const char* help_string_save_only           = "only perform load/save phase"                    ;
static const char* help_string_write_input         = "write to files named \"input-#\".bin"            ;
static const char* help_string_write_output        = "write to files named \"output-#\".bin"           ;
static const char* help_string_gen_golden          = "write to files named \"golden-#\".bin"           ;
static const char* help_string_gen_state           = "process json file to dd_metastate files"         ;
static const char* help_string_use_state           = "ignore json file"                                ;
static const char* help_string_compare_output      = "compare output to files named  \"golden-#\".bin" ;
static const char* help_string_no_execute          = "terminate before running model"                  ;
static const char* help_string_no_avoid            = "always run test (ignore \"avoid\" file)"         ;
static const char* help_string_local_xclbin        = "use xclbin in model folder, instead of DD_ROOT"  ;
static const char* help_string_test_configs        = "test after save_state with different  configs"   ;
static const char* help_string_test_config         = "test specific config"                            ;
static const char* help_string_init_method         = "how to initialize input"                         ;
static const char* help_string_print_summary       = "print metastate info"                            ;
static const char* help_string_print_perf          = "print perf info"                                 ;
static const char* help_string_print_debug         = "print debug info"                                ;
static const char* help_string_quiet               = "be quiet"                                        ;
static const char* help_string_cleanup             = "delete files generated for test_configs"         ;

static const char* string_optimize_scratch    = "optimize_scratch"    ;
static const char* string_eager_mode          = "eager_mode"          ;
static const char* string_use_lazy_scratch_bo = "use_lazy_scratch_bo" ;
static const char* string_en_lazy_constbo     = "en_lazy_constbo"     ;
static const char* string_dealloc_scratch_bo  = "dealloc_scratch_bo"  ;
static const char* string_disable_preemption  = "disable_preemption"  ;

static const char* short_string_optimize_scratch    = "optsbo"  ;
static const char* short_string_eager_mode          = "eager"   ;
static const char* short_string_use_lazy_scratch_bo = "lazysbo" ;
static const char* short_string_en_lazy_constbo     = "lazycbo" ;
static const char* short_string_dealloc_scratch_bo  = "freesbo" ;
static const char* short_string_disable_preemption  = "nopre"   ;

static const char* help_string_optimize_scratch    = "compile-time"   ;
static const char* help_string_eager_mode          = "compile-time"   ;
static const char* help_string_use_lazy_scratch_bo = "run-time"       ;
static const char* help_string_en_lazy_constbo     = "run-time"       ;
static const char* help_string_dealloc_scratch_bo  = "run-time"       ;
static const char* help_string_disable_preemption  = "compile-time"   ;

static const command_short_help CSH_ACTIONS[] = {
  { string_write_input   , short_string_write_input    , help_string_write_input    , "I"  },
  { string_write_output  , short_string_write_output   , help_string_write_output   , "O"  },
  { string_gen_golden    , short_string_gen_golden     , help_string_gen_golden     , "G"  },
  { string_gen_state     , short_string_gen_state      , help_string_gen_state      , "GS" },
  { string_compare_output, short_string_compare_output , help_string_compare_output , "C"  },
  { }
};

static const command_short_help CSH_OPTIONS[] = {
  { string_load_only     , short_string_load_only      , help_string_load_only      , "L"  },
  { string_save_only     , short_string_save_only      , help_string_save_only      , "S"  },
  { string_no_execute    , short_string_no_execute     , help_string_no_execute     , "X"  },
  { string_no_avoid      , short_string_no_avoid       , help_string_no_avoid       , "A"  },
  { string_local_xclbin  , short_string_local_xclbin   , help_string_local_xclbin   , "M"  },
  { string_test_configs  , short_string_test_configs   , help_string_test_configs   , "T"  },
  { string_test_config   , short_string_test_config    , help_string_test_config    , "T#" },
  { string_print_summary , short_string_print_summary  , help_string_print_summary  , "V"  },
  { string_print_perf    , short_string_print_perf     , help_string_print_perf     , "P"  },
  { string_print_debug   , short_string_print_debug    , help_string_print_debug    , "D"  },
  { string_quiet         , short_string_quiet          , help_string_quiet          , "Q"  },
  { }
};

static const command_short_help CSH_DDOPTIONS[] = {
  { string_optimize_scratch    , short_string_optimize_scratch    , help_string_optimize_scratch    , "1" },
  { string_eager_mode          , short_string_eager_mode          , help_string_eager_mode          , "2" },
  { string_use_lazy_scratch_bo , short_string_use_lazy_scratch_bo , help_string_use_lazy_scratch_bo , "3" },
  { string_en_lazy_constbo     , short_string_en_lazy_constbo     , help_string_en_lazy_constbo     , "4" },
  { string_dealloc_scratch_bo  , short_string_dealloc_scratch_bo  , help_string_dealloc_scratch_bo  , "5" },
  { string_disable_preemption  , short_string_disable_preemption  , help_string_disable_preemption  , "6" },
  { }
};


static void print_CSH_ACTIONS(std::ostream &ostr, const char *indent, bool no_header) {
  if (!no_header) {
    ostr
      << indent << "These are the recognized actions:"    << std::endl
      << std::endl;
  }
  const TMF::command_short_help *csh = TMF::CSH_ACTIONS;
  while (csh->command_string) {
    ostr << indent;
    csh->print(ostr, 20, 50, 5, 3);
    ostr << std::endl;
    ++csh;
  }
  std::cout
    << std::endl;
}

static void print_CSH_OPTIONS(std::ostream &ostr, const char *indent, bool no_header) {
  if (!no_header) {
    ostr
      << indent << "These are the recognized options:"    << std::endl
      << std::endl;
  }
  const TMF::command_short_help *csh = TMF::CSH_OPTIONS;
  while (csh->command_string) {
    ostr << indent;
    csh->print(ostr, 20, 50, 5, 3);
    ostr << std::endl;
    ++csh;
  }
  std::cout
    << std::endl;
}

static void print_CSH_DDOPTIONS(std::ostream &ostr, const char *indent, bool no_header) {
  if (!no_header) {
    ostr
      << indent << "The \"" << TMF::string_test_config << "\" option expects the name of a DDOptions field:"    << std::endl
      << std::endl;
  }
  const TMF::command_short_help *csh = TMF::CSH_DDOPTIONS;
  while (csh->command_string) {
    ostr << indent;
    csh->print(ostr, 20, 46, 8, 3);
    ostr << std::endl;
    ++csh;
  }
  std::cout
    << std::endl;
}

static void print_USAGE(std::ostream &ostr, const char *cmdname) {
  ostr
    << "Usage : " << cmdname << " " << "meta.json    meta.state  "   "file.xclbin {actions} [options] [niters=1]"    << std::endl
    << "        " << cmdname << " " << "foldername [..]"                                                             << std::endl
    << "        " << cmdname << " " << "--use_state  meta.state  "   "file.xclbin [..]"                              << std::endl
    << "        " << cmdname << " " << "meta.json   --gen_state  "   "file.xclbin --test_configs [..]"               << std::endl
    << "        " << cmdname << " " << "meta.json   --gen_state  "   "file.xclbin --test_config \"configstr\" [..]"  << std::endl
    << "        " << cmdname << " " << "meta.json   --gen_state  "   "file.xclbin [--save_only]"                     << std::endl
    << std::endl;
}

static void print_WARNING(std::ostream &ostr, const char *indent) {
  ostr
    << std::endl
    << "WARNING: files named \"" << TMF::string_dd_metastate << "\" will be overwritten if the command line parameter \"meta.state\" is an empty string."        << std::endl
    << indent << "When the \"meta.state\" parameter is an empty string, the json file will be used to create \"" << TMF::string_dd_metastate << "*.*\" files."   << std::endl
    << indent << "Files named \"" << TMF::string_dd_metastate << "*.*\" will be overwritten without prompting."                                                  << std::endl
    << indent << "Use \"" << TMF::string_load_only      << "\" to prevent saving files."                                                                         << std::endl
    << std::endl;
}

// clang-format on
