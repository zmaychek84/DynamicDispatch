/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#ifndef SD_HELPER_HPP
#define SD_HELPER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utils/logging.hpp>
#include <vector>

namespace ryzenai {
namespace sd_helper {

// Helper functions
static std::string int2hex(int64_t n, int bits) {
  int len = static_cast<int>(std::ceil(bits / 4.0) + 2);
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  if (n >= 0) {
    ss << "0x" << std::setw(len - 2) << n;
  } else {
    uint64_t val = (1ULL << bits) + n;
    ss << "0x" << std::setw(len - 2) << val;
  }
  std::string o = ss.str();
  return o.substr(2);
}

static uint32_t floatToBits(float f) {
  uint32_t result;
  std::memcpy(&result, &f, sizeof(f));
  return result;
}

static std::string convert_float_to_hex(const std::string &in_line) {
  std::istringstream iss(in_line);
  std::vector<std::string> arr;
  std::string temp;
  while (iss >> temp) {
    arr.push_back(temp);
  }
  size_t num_elems = arr.size();
  std::string out_hex = "";
  for (int i = 0; i < num_elems; ++i) {
    float s = std::stof(arr[num_elems - 1 - i]);
    uint32_t s_int = floatToBits(s);
    out_hex += int2hex(static_cast<int32_t>(s_int), 32);
  }
  return out_hex + '\n';
}

static std::string convert_int8_to_hex(const std::string &in_line) {
  std::istringstream iss(in_line);
  std::vector<std::string> arr;
  std::string temp;
  while (iss >> temp) {
    arr.push_back(temp);
  }
  size_t num_elems = arr.size();
  std::string out_hex = "";
  for (int i = 0; i < num_elems; ++i) {
    int8_t s = static_cast<int8_t>(std::stoi(arr[num_elems - 1 - i]));
    out_hex += int2hex(static_cast<int32_t>(s), 8);
  }
  return out_hex + '\n';
}

static void aie_srs(std::vector<uint32_t> &input_output) {
  int data_width = 16;
  int shift = 16;
  for (size_t i = 0; i < input_output.size(); ++i) {
    float temp =
        float(static_cast<float>(input_output[i]) / std::pow(2.0f, shift));
    temp = std::round(temp);
    if (temp > std::pow(2.0f, data_width) - 1) {
      temp = float(std::pow(2.0f, data_width)) - 1;
    }
    if (temp < 0) {
      temp = 0;
    }
    input_output[i] = static_cast<uint32_t>(temp);
  }
}

static void float2bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs(x_uint32);
  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint32[i] = (static_cast<uint16_t>(x_uint32[i]) << 16);
  }
  std::memcpy(x.data(), x_uint32.data(), x.size() * sizeof(float));
}

static std::vector<uint8_t> compress_bf16(const std::vector<uint32_t> &data,
                                          int block_size, int sub_block_size,
                                          int sub_block_shift_bits) {
  int m_bfp = 16 - 9;
  int exp_bias = 127;
  std::vector<uint8_t> ret(block_size + 1);
  std::vector<uint8_t> exp_data(block_size);
  for (int i = 0; i < block_size; ++i) {
    uint8_t exp = (data[i] & 0x7F800000) >> 23;
    exp_data[i] = exp;
  }
  uint8_t shared_exp = *std::max_element(exp_data.begin(), exp_data.end());
  ret[0] = shared_exp;

  for (int i = 0; i < block_size / sub_block_size; ++i) {
    uint8_t max_sub_exp =
        *std::max_element(exp_data.begin() + i * sub_block_size,
                          exp_data.begin() + (i + 1) * sub_block_size);
    int shift_upper_bound = (1 << sub_block_shift_bits) - 1;
    int shift =
        std::min(static_cast<int>(shared_exp - max_sub_exp), shift_upper_bound);

    for (int j = 0; j < sub_block_size; ++j) {
      uint32_t fp32_data = data[i * sub_block_size + j];
      float bf16_data;
      std::memcpy(&bf16_data, &fp32_data, sizeof(float));
      float sign_mantissa =
          bf16_data /
          std::pow(2.0f, (shared_exp - exp_bias - shift + 1.0f - m_bfp));
      ret[i * sub_block_size + j + 1] = static_cast<int8_t>(sign_mantissa);
    }
  }
  return ret;
}

static std::vector<float> read_binary_file(const std::string &filename) {
  std::ifstream infile(filename, std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return {};
  }

  std::streamsize file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  size_t num_elements = file_size / sizeof(float);
  std::vector<float> array(num_elements);
  if (infile.read(reinterpret_cast<char *>(array.data()), file_size)) {
    std::cout << "Successfully read " << num_elements << " elements from "
              << filename << std::endl;
  } else {
    std::cerr << "Error reading file!" << std::endl;
    array.clear();
  }
  infile.close();
  return array;
}

// Class declaration for Range
class Range {
public:
  size_t start;
  size_t end;
  Range(size_t s = 0, size_t e = std::numeric_limits<size_t>::max())
      : start(s), end(e) {}
};

// Tensor class template
template <typename T> class Tensor {
private:
  std::vector<T> data;
  size_t dim0, dim1, dim2, dim3;

public:
  Tensor(size_t d0 = 0, size_t d1 = 0, size_t d2 = 0, size_t d3 = 0);
  Tensor(const std::vector<T> &input_data, const std::vector<size_t> &shape);
  Tensor(const Tensor<T> &other);
  Tensor<T> &operator=(const Tensor<T> &other);
  T &operator()(size_t i, size_t j, size_t k, size_t l);
  const T &operator()(size_t i, size_t j, size_t k, size_t l) const;
  Tensor<T> slice(Range r0 = Range(), Range r1 = Range(), Range r2 = Range(),
                  Range r3 = Range()) const;
  size_t size(size_t dim) const;
  void print_shape() const;
  const std::vector<T> &get_data() const;
};

// Implement Tensor class template functions
template <typename T>
Tensor<T>::Tensor(size_t d0, size_t d1, size_t d2, size_t d3)
    : dim0(d0), dim1(d1), dim2(d2), dim3(d3), data(d0 * d1 * d2 * d3) {}

template <typename T>
Tensor<T>::Tensor(const std::vector<T> &input_data,
                  const std::vector<size_t> &shape) {
  dim0 = shape.size() > 0 ? shape[0] : 1;
  dim1 = shape.size() > 1 ? shape[1] : 1;
  dim2 = shape.size() > 2 ? shape[2] : 1;
  dim3 = shape.size() > 3 ? shape[3] : 1;

  size_t total_size = dim0 * dim1 * dim2 * dim3;
  if (input_data.size() != total_size) {
    throw std::invalid_argument(
        "Data size does not match the shape's total size");
  }

  data = input_data;
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T> &other)
    : dim0(other.dim0), dim1(other.dim1), dim2(other.dim2), dim3(other.dim3),
      data(other.data) {}

template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
  if (this != &other) {
    dim0 = other.dim0;
    dim1 = other.dim1;
    dim2 = other.dim2;
    dim3 = other.dim3;
    data = other.data;
  }
  return *this;
}

template <typename T>
T &Tensor<T>::operator()(size_t i, size_t j, size_t k, size_t l) {
  if (i >= dim0 || j >= dim1 || k >= dim2 || l >= dim3) {
    throw std::out_of_range("Tensor index out of range");
  }
  size_t index = ((i * dim1 + j) * dim2 + k) * dim3 + l;
  return data[index];
}

template <typename T>
const T &Tensor<T>::operator()(size_t i, size_t j, size_t k, size_t l) const {
  if (i >= dim0 || j >= dim1 || k >= dim2 || l >= dim3) {
    throw std::out_of_range("Tensor index out of range");
  }
  size_t index = ((i * dim1 + j) * dim2 + k) * dim3 + l;
  return data[index];
}

template <typename T>
Tensor<T> Tensor<T>::slice(Range r0, Range r1, Range r2, Range r3) const {
  size_t s0_start = r0.start;
  size_t s0_end =
      (r0.end == std::numeric_limits<size_t>::max() || r0.end > dim0) ? dim0
                                                                      : r0.end;
  size_t s1_start = r1.start;
  size_t s1_end =
      (r1.end == std::numeric_limits<size_t>::max() || r1.end > dim1) ? dim1
                                                                      : r1.end;
  size_t s2_start = r2.start;
  size_t s2_end =
      (r2.end == std::numeric_limits<size_t>::max() || r2.end > dim2) ? dim2
                                                                      : r2.end;
  size_t s3_start = r3.start;
  size_t s3_end =
      (r3.end == std::numeric_limits<size_t>::max() || r3.end > dim3) ? dim3
                                                                      : r3.end;

  if (s0_end > dim0 || s1_end > dim1 || s2_end > dim2 || s3_end > dim3) {
    throw std::out_of_range("Slice range out of bounds");
  }

  size_t new_dim0 = s0_end - s0_start;
  size_t new_dim1 = s1_end - s1_start;
  size_t new_dim2 = s2_end - s2_start;
  size_t new_dim3 = s3_end - s3_start;

  Tensor<T> result(new_dim0, new_dim1, new_dim2, new_dim3);
  for (size_t i = 0; i < new_dim0; ++i) {
    for (size_t j = 0; j < new_dim1; ++j) {
      for (size_t k = 0; k < new_dim2; ++k) {
        for (size_t l = 0; l < new_dim3; ++l) {
          result(i, j, k, l) =
              (*this)(i + s0_start, j + s1_start, k + s2_start, l + s3_start);
        }
      }
    }
  }

  return result;
}

template <typename T> size_t Tensor<T>::size(size_t dim) const {
  switch (dim) {
  case 0:
    return dim0;
  case 1:
    return dim1;
  case 2:
    return dim2;
  case 3:
    return dim3;
  default:
    throw std::out_of_range("Invalid dimension");
  }
}

template <typename T> void Tensor<T>::print_shape() const {
  std::cout << "Tensor shape: (" << dim0 << ", " << dim1 << ", " << dim2 << ", "
            << dim3 << ")\n";
}

template <typename T> const std::vector<T> &Tensor<T>::get_data() const {
  return data;
}

// Helper function to convert Tensor to a vector of uint32_t (for compression)
template <typename T>
std::vector<uint32_t> tensor_to_vector(const Tensor<T> &tensor) {
  size_t num_elements =
      tensor.size(0) * tensor.size(1) * tensor.size(2) * tensor.size(3);
  std::vector<uint32_t> result(num_elements);
  auto data = tensor.get_data();
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = *reinterpret_cast<const uint32_t *>(&data[i]);
  }
  return result;
}

struct layer_params {
  uint32_t ifm_sv_width;
  uint32_t ifm_sv_height;
  uint32_t ifm_sv_depth;
  uint32_t ofm_sv_depth;
  uint32_t kwidth;
  uint32_t kheight;
  uint32_t channel_mode;
  uint32_t stride_bit;
  uint32_t psum_a_offset;
  uint32_t lrelu_alpha_kernel;
  uint32_t act;
  uint32_t w_loop;
  uint32_t h_loop;
  uint32_t oc_loop;
  uint32_t ic_loop;
  uint32_t ofm_sv_width;
  uint32_t ifm_depth_iters;
  uint32_t num_wt_streams;
  uint32_t casc_len;
  uint32_t ifm_depth, ofm_depth;

  layer_params()
      : ifm_sv_width(32), ifm_sv_height(1), ifm_sv_depth(96), ofm_sv_depth(80),
        kwidth(1), kheight(1), channel_mode(0), stride_bit(0), psum_a_offset(0),
        lrelu_alpha_kernel(0), act(0), w_loop(1), h_loop(8), oc_loop(2),
        ic_loop(20), ofm_sv_width(32), num_wt_streams(8), ifm_depth(1920),
        ofm_depth(640), casc_len(1) {
    ifm_depth_iters = (uint32_t)std::ceil(ifm_depth / ifm_sv_depth);
  }

  layer_params(const uint32_t *lp_data_ptr, uint32_t ifm_depth_val,
               uint32_t ofm_depth_val) {
    ifm_sv_width = lp_data_ptr[0];
    ifm_sv_height = lp_data_ptr[1];
    ifm_sv_depth = lp_data_ptr[2];
    ofm_sv_depth = lp_data_ptr[3];
    kwidth = lp_data_ptr[4];
    kheight = lp_data_ptr[5];
    channel_mode = lp_data_ptr[6];
    stride_bit = lp_data_ptr[7];
    psum_a_offset = lp_data_ptr[8];
    lrelu_alpha_kernel = lp_data_ptr[9];
    act = lp_data_ptr[10];
    w_loop = lp_data_ptr[11];
    h_loop = lp_data_ptr[12];
    // ofm_depth_iters in python
    oc_loop = lp_data_ptr[13];
    ic_loop = lp_data_ptr[14];
    ofm_sv_width = lp_data_ptr[15];
    ifm_depth = ifm_depth_val;
    ofm_depth = ofm_depth_val;

    ifm_depth_iters = (uint32_t)std::ceil(float(ifm_depth) / ifm_sv_depth);
    num_wt_streams =
        (uint32_t)std::ceil(float(ofm_depth) / (oc_loop * ofm_sv_depth));
    // hardcode for now
    casc_len = 1;
  }

  void print_param() const {
    RYZENAI_LOG_TRACE("IFM SV Width: " + std::to_string(ifm_sv_width));
    RYZENAI_LOG_TRACE("IFM SV Height: " + std::to_string(ifm_sv_height));
    RYZENAI_LOG_TRACE("IFM SV Depth: " + std::to_string(ifm_sv_depth));
    RYZENAI_LOG_TRACE("OFM SV Depth: " + std::to_string(ofm_sv_depth));
    RYZENAI_LOG_TRACE("Kernel Width: " + std::to_string(kwidth));
    RYZENAI_LOG_TRACE("Kernel Height: " + std::to_string(kheight));
    RYZENAI_LOG_TRACE("Channel Mode: " + std::to_string(channel_mode));
    RYZENAI_LOG_TRACE("Stride Bit: " + std::to_string(stride_bit));
    RYZENAI_LOG_TRACE("PSUM A Offset: " + std::to_string(psum_a_offset));
    RYZENAI_LOG_TRACE("LeakyReLU Alpha Kernel: " +
                      std::to_string(lrelu_alpha_kernel));
    RYZENAI_LOG_TRACE("Activation: " + std::to_string(act));
    RYZENAI_LOG_TRACE("W Loop: " + std::to_string(w_loop));
    RYZENAI_LOG_TRACE("H Loop: " + std::to_string(h_loop));
    RYZENAI_LOG_TRACE("OC Loop: " + std::to_string(oc_loop));
    RYZENAI_LOG_TRACE("IC Loop: " + std::to_string(ic_loop));
    RYZENAI_LOG_TRACE("OFM SV Width: " + std::to_string(ofm_sv_width));
    RYZENAI_LOG_TRACE("IFM Depth Iters: " + std::to_string(ifm_depth_iters));
    RYZENAI_LOG_TRACE("Num WTS Streams: " + std::to_string(num_wt_streams));
    RYZENAI_LOG_TRACE("CASC Len: " + std::to_string(casc_len));
    RYZENAI_LOG_TRACE("IFM Depth: " + std::to_string(ifm_depth));
    RYZENAI_LOG_TRACE("OFM Depth: " + std::to_string(ofm_depth));
  }
};

// preserve high precision
static std::string float_to_string(float value) {
  std::ostringstream oss;
  oss << std::setprecision(12) << value; // Using 12 significant digits
  return oss.str();
}

template <typename T>
void write_datafmt_wts(std::vector<uint32_t> &buffer,
                       const std::vector<T> &wts_data_vec,
                       const std::vector<size_t> &wt_shape,
                       const std::vector<float> &bias_data,
                       const layer_params &lp, const std::string &fname,
                       size_t CStride = 8, int num_words_bias = 1,
                       int num_words_wts = 2, bool write_to_file = false) {

  if constexpr (!std::is_same<T, float>::value) {
    throw std::runtime_error(
        "Template parameter T for write_datafmt_wts must be float!");
  }

  Tensor<T> wt_data(wts_data_vec, wt_shape);
  size_t ifm_sv_depth = lp.ifm_sv_depth;
  size_t ofm_sv_depth = lp.ofm_sv_depth;
  size_t ifm_depth_iters = lp.ifm_depth_iters;
  size_t num_wt_streams = lp.num_wt_streams;
  size_t ofm_depth_iters = lp.oc_loop;
  size_t casc_len = lp.casc_len;
  size_t cout_per_ch_iter = wt_shape[0] / ofm_depth_iters;
  size_t cout_per_stream = cout_per_ch_iter / num_wt_streams;
  size_t depth_iter_casc = ifm_depth_iters / casc_len;

  std::vector<Tensor<T>> wts_list;
  std::vector<std::vector<float>> bias;

  for (int csc_len = 0; csc_len < casc_len; ++csc_len) {
    for (int och_iter = 0; och_iter < ofm_depth_iters; ++och_iter) {
      Tensor<T> wt_och_iter = wt_data.slice(
          Range(och_iter * cout_per_ch_iter, (och_iter + 1) * cout_per_ch_iter),
          Range(), Range(), Range());
      std::vector<float> b_och_iter(
          bias_data.begin() + och_iter * cout_per_ch_iter,
          bias_data.begin() + (och_iter + 1) * cout_per_ch_iter);
      for (int wt_strms = 0; wt_strms < num_wt_streams; ++wt_strms) {
        Tensor<T> wt_strm_data = wt_och_iter.slice(
            Range(wt_strms * cout_per_stream, (wt_strms + 1) * cout_per_stream),
            Range(), Range(), Range());
        std::vector<float> b_strm_data(
            b_och_iter.begin() + wt_strms * cout_per_stream,
            b_och_iter.begin() + (wt_strms + 1) * cout_per_stream);
        for (int ich_iter = 0; ich_iter < depth_iter_casc; ++ich_iter) {
          Tensor<T> wts_temp = wt_strm_data.slice(
              Range(),
              Range(ich_iter * ifm_sv_depth, (ich_iter + 1) * ifm_sv_depth),
              Range(), Range());
          wts_list.push_back(wts_temp);
          if (csc_len == 0) {
            bias.push_back(b_strm_data);
          }
        }
      }
    }
  }

  std::ofstream wts32_fp;
  if (write_to_file) {
    wts32_fp.open(fname);
  }

  for (size_t it = 0; it < wts_list.size(); ++it) {
    Tensor<T> &wts = wts_list[it];
    int id = 0;
    std::string plio_line = "";

    std::vector<float> b_vals;
    for (size_t o = 0; o < ofm_sv_depth; o += CStride) {
      for (size_t repeat = 0; repeat < 4; ++repeat) {
        std::vector<float> b(bias[it].begin() + o,
                             bias[it].begin() + o + CStride);
        for (size_t i = 0; i < CStride; ++i) {
          b_vals.push_back(b[i]);
        }
      }
    }
    id = 0;
    plio_line = "";
    for (size_t i = 0; i < b_vals.size(); ++i) {
      if ((id + 1) % num_words_bias == 0) {
        buffer.push_back(*reinterpret_cast<const uint32_t *>(&b_vals[i]));
        plio_line += float_to_string(b_vals[i]) + '\n';
        if (write_to_file) {
          wts32_fp << convert_float_to_hex(plio_line);
        }
        plio_line = "";
      } else {
        plio_line += std::to_string(b_vals[i]) + ' ';
      }
      id++;
    }
    size_t Cout = wts.size(0);
    size_t Cin = wts.size(1);
    size_t Kx = wts.size(3);
    size_t Ky = wts.size(2);
    size_t istride = std::min(Cin, CStride);
    std::vector<uint8_t> w_vals;
    size_t ostride = 16;

    for (size_t o = 0; o < Cout; o += ostride) {
      for (size_t i = 0; i < Cin; i += istride) {
        for (size_t y = 0; y < Ky; ++y) {
          for (size_t x = 0; x < Kx; ++x) {
            for (size_t o_idx = o; o_idx < o + 8; ++o_idx) {
              Tensor<T> w =
                  wts.slice(Range(o_idx, o_idx + 1), Range(i, i + istride),
                            Range(y, y + 1), Range(x, x + 1));
              std::vector<uint8_t> compressed_w =
                  compress_bf16(tensor_to_vector(w), 8, 8, 0);
              w_vals.insert(w_vals.end(), compressed_w.begin(),
                            compressed_w.end());
            }

            for (size_t o_idx = o + 8; o_idx < o + 16; ++o_idx) {
              Tensor<T> w =
                  wts.slice(Range(o_idx, o_idx + 1), Range(i, i + istride),
                            Range(y, y + 1), Range(x, x + 1));
              std::vector<uint8_t> compressed_w =
                  compress_bf16(tensor_to_vector(w), 8, 8, 0);
              w_vals.insert(w_vals.end(), compressed_w.begin(),
                            compressed_w.end());
            }
          }
        }
      }
    }

    id = 0;
    plio_line = "";
    uint32_t result = 0;
    for (size_t i = 0; i < w_vals.size(); ++i) {
      size_t id4 = i % 4;
      result |= (static_cast<uint32_t>(w_vals[i]) & 0xFF) << (id4 * 8);
      if ((id + 1) % 4 == 0) {
        plio_line += std::to_string(static_cast<int>(w_vals[i])) + '\n';
        if (write_to_file) {
          wts32_fp << convert_int8_to_hex(plio_line);
        }
        buffer.push_back(result);
        result = 0;
        plio_line = "";
      } else {
        plio_line += std::to_string(static_cast<int>(w_vals[i])) + ' ';
      }
      id++;
    }
    if (w_vals.size() % 4) {
      buffer.push_back(result);
    }
  }
  if (write_to_file) {
    wts32_fp.close();
  }
}

static void save_result_to_hex_file(const std::vector<uint8_t> &result,
                                    const std::string &file_name) {
  std::ofstream file(file_name);

  if (file.is_open()) {
    size_t num_uint32 = result.size() / 4;

    const uint32_t *uint32_data =
        reinterpret_cast<const uint32_t *>(result.data());

    for (size_t i = 0; i < num_uint32; ++i) {
      file << int2hex(uint32_data[i], 32) << std::endl;
    }
    size_t remaining_bytes = result.size() % 4;
    if (remaining_bytes > 0) {
      uint32_t last_value = 0;
      const uint8_t *last_bytes = &result[num_uint32 * 4];
      std::memcpy(&last_value, last_bytes, remaining_bytes);
      file << int2hex(last_value, 32) << std::endl;
    }

    file.close();
    std::cout << "Result saved to " << file_name << " in hexadecimal format."
              << std::endl;
  } else {
    std::cerr << "Failed to open file: " << file_name << std::endl;
  }
}

static void shuffle_gemm_wts(std::vector<uint8_t> &w_b_vals, const float *input,
                             int rows, int cols, bool wts_transpose,
                             const float *bias, int sv_k, int sv_n) {
  int iter_k = rows / sv_k;
  int iter_n = cols / sv_n;

  auto elem_sz_per_line = 2;

  if (wts_transpose) {
    // Transpose the input matrix
    std::vector<float> input_transpose(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        input_transpose[j * rows + i] = input[i * cols + j];
      }
    }

    int sv_k_div8 = sv_k / 8;
    for (int n = 0; n < iter_n; ++n) {
      for (int k = 0; k < iter_k; ++k) {
        // handle bias
        if (bias != nullptr) {
          std::vector<uint16_t> bias_sv_shuffled;
          std::vector<float> lst;
          int start_idx = n * sv_n;
          int end_idx = start_idx + sv_n;
          std::vector<float> bias_sv(bias + start_idx, bias + end_idx);
          for (size_t i = 0; i < bias_sv.size(); ++i) {

            lst.push_back(bias_sv[i]);
            if ((i + 1) % elem_sz_per_line == 0) {
              uint32_t float_data;
              std::memcpy(&float_data, &lst[0], sizeof(float));
              uint16_t bf16_data = float_data >> 16;
              bias_sv_shuffled.push_back(bf16_data);

              std::memcpy(&float_data, &lst[1], sizeof(float));
              bf16_data = float_data >> 16;
              bias_sv_shuffled.push_back(bf16_data);
              lst.clear();
            }
          }
          uint8_t *bias_8bit =
              reinterpret_cast<uint8_t *>(bias_sv_shuffled.data());
          w_b_vals.insert(w_b_vals.end(), bias_8bit,
                          bias_8bit + bias_sv_shuffled.size() * 2);
        }
        // handle weight
        std::vector<uint8_t> w_vals;
        for (int r = 0; r < sv_k_div8; ++r) {
          for (int c = 0; c < sv_n; ++c) {
            float data[8];
            for (int idx = 0; idx < 8; ++idx) {
              int row = n * sv_n + c;
              int col = k * sv_k + r * 8 + idx;
              if (row < cols && col < rows) {
                data[idx] = input_transpose[row * rows + col];
              } else {
                data[idx] = 0.0f; // Zero padding if necessary
              }
            }
            std::vector<uint32_t> uint32_data(8);
            std::memcpy(uint32_data.data(), data, 8 * sizeof(float));
            std::vector<uint8_t> w = compress_bf16(uint32_data, 8, 8, 0);
            w_b_vals.insert(w_b_vals.end(), w.begin(), w.end());
          }
        }
      }
    }
  } else {
    // Handle the case where wts_transpose == false if necessary
  }
  // may use save_result_to_hex_file to dump the shuffled data w_b_vals.
  return;
}

} // namespace sd_helper

} // namespace ryzenai

#endif // SD_HELPER_HPP
