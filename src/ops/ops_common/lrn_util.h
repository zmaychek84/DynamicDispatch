/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __LRN_UTIL_H__
#define __LRN_UTIL_H__

#include <fstream>
#include <iostream>
#include <string>

class LrnUtil {
public:
  LrnUtil();
  void Float_Bits_to_INT8(int8_t *store, float a);
  bool get_input_from_file_bf16(int8_t *in_subv, std::string file);
  // bool get_parameter_input_bf16(int8_t* params_ptr, std::vector<std::string>
  // param_files);
  bool get_input_from_file_int8(int8_t *input, std::string file);
  float get_maximum_difference_BF16(int8_t *output, int8_t *output_reference,
                                    int number_of_values);
  int number_of_rounding_errors(int8_t *output, int8_t *output_reference,
                                int number_of_values);
  bool within_delta(int8_t *output, int8_t *reference, int number_of_values);
  bool write_to_file_bf16(int8_t *output_subv, std::string file,
                          int num_elements);
  bool write_to_file_int8(int8_t *output_subv, std::string file,
                          int num_elements);
};

#include "lrn_util.cpp"

#endif
