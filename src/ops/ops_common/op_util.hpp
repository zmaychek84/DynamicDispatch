#pragma once

#include <any>
#include <iostream>
#include <map>
#include <ops/op_interface.hpp>
#include <utils/tfuncs.hpp>
#include <vector>
namespace OpsFusion {

static bool check_generic_fusion(const std::map<std::string, std::any> &attr) {
  if (attr.count("generic_fusion")) {
    return true;
  }
  return false;
}

static bool check_bias(const std::map<std::string, std::any> &attr) {
  if (attr.count("bias")) {
    return true;
  }
  return false;
}

static bool check_gelu(const std::map<std::string, std::any> &attr) {
  if (attr.count("gelu")) {
    return true;
  }
  return false;
}

static size_t reduce(const std::vector<size_t> &shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), (size_t)1,
                         std::multiplies<size_t>{});
}

static std::vector<std::vector<int64_t>>
stringsToVector2d(const std::vector<std::string> &vecStr) {
  std::vector<std::vector<int64_t>> result;
  result.reserve(vecStr.size());

  for (const auto &str : vecStr) {
    // Validate format
    if (str.empty() || str[0] != '[' || str.back() != ']') {
      throw std::invalid_argument(
          "Invalid string format. Expected '[num1,num2,...]': " + str);
    }

    std::vector<int64_t> innerVec;
    std::istringstream stream(
        str.substr(1, str.size() - 2)); // Remove '[' and ']'
    std::string numStr;

    while (std::getline(stream, numStr, ',')) {
      try {
        if (!numStr.empty()) {
          innerVec.push_back(std::stoll(numStr));
        }
      } catch (const std::exception &) {
        throw std::invalid_argument("Invalid number format in '" + numStr +
                                    "' in string: " + str);
      }
    }

    result.push_back(std::move(innerVec));
  }

  return result;
}

template <typename T> static std::vector<std::vector<T>> fold2D(Tensor ws) {
  //   CHECK(ws.size() == (size_t)reduce(shape))
  //       << ws.size() << "!=" << (size_t)reduce(shape);
  DD_THROW_IF(ws.shape.size() != 2, "Only 2D tensors are supported");
  size_t size = reduce(ws.shape);
  // RYZENAI_LOG_TRACE("SB SIZE= " + size);
  int32_t rows = (int32_t)ws.shape[0];
  int32_t cols = (int32_t)ws.shape[1];
  std::vector<T> src(reinterpret_cast<T *>(ws.data),
                     reinterpret_cast<T *>(ws.data) + size);
  std::vector<std::vector<T>> ret(rows);
  for (int i = 0; i < rows; ++i) {
    ret[i].resize(cols);
  }

  for (size_t i = 0; i < src.size(); ++i) {
    int r = (int)i / cols;
    int c = (int)i % cols;
    ret[r][c] = src[i];
  }
  return ret;
}

template <typename T>
static std::vector<std::vector<std::vector<T>>> fold3D(Tensor t) {
  const T *ws = (T *)t.data;
  const std::vector<size_t> shape = t.shape;

  if (shape.size() != 3) {
    throw std::runtime_error("[ERROR] : Shape size should be 3.");
  }

  int32_t batches = (int32_t)shape[0];
  int32_t rows = (int32_t)shape[1];
  int32_t cols = (int32_t)shape[2];

  size_t size =
      batches * rows * cols; // Calculate total size from shape dimensions

  std::vector<std::vector<std::vector<T>>> ret(batches);
  for (int n = 0; n < batches; ++n) {
    ret[n].resize(rows);
    for (int m = 0; m < rows; ++m) {
      ret[n][m].resize(cols);
    }
  }

  for (size_t i = 0; i < size; ++i) {
    int b = (int)i / (cols * rows);
    int r = (int)(i - b * cols * rows) / cols;
    int c = (int)i % cols;
    ret[b][r][c] = ws[i];
  }
  return ret;
}

static std::vector<uint16_t> get_tensor_as_uint16_t_vec(Tensor ws) {
  size_t size = reduce(ws.shape);
  if (ws.dtype == "uint8") {
    std::vector<uint8_t> src(reinterpret_cast<uint8_t *>(ws.data),
                             reinterpret_cast<uint8_t *>(ws.data) + size);
    std::vector<uint16_t> r(src.begin(), src.end());
    return r;
  } else if (ws.dtype == "uint16") {
    std::vector<uint16_t> r(reinterpret_cast<uint16_t *>(ws.data),
                            reinterpret_cast<uint16_t *>(ws.data) + size);
    return r;
  }
  throw std::runtime_error(
      "Other than uint8 and uint16, format not supported. Received : " +
      ws.dtype);
}

static std::vector<int64_t> get_tensor_as_int64_t_vec(Tensor ws) {
  size_t size = reduce(ws.shape);
  if (ws.dtype == "uint8") {
    std::vector<uint8_t> src(reinterpret_cast<uint8_t *>(ws.data),
                             reinterpret_cast<uint8_t *>(ws.data) + size);
    std::vector<int64_t> r(src.begin(), src.end());
    return r;
  } else if (ws.dtype == "uint16") {
    std::vector<uint16_t> src(reinterpret_cast<uint16_t *>(ws.data),
                              reinterpret_cast<uint16_t *>(ws.data) + size);
    std::vector<int64_t> r(src.begin(), src.end());
    return r;
  } else if (ws.dtype == "int32") {
    std::vector<int32_t> src(reinterpret_cast<int32_t *>(ws.data),
                             reinterpret_cast<int32_t *>(ws.data) + size);
    std::vector<int64_t> r(src.begin(), src.end());
    return r;
  }
  throw std::runtime_error(
      "Other than int32, uint8 and uint16, format not supported. Received : " +
      ws.dtype);
}

static std::vector<float> get_tensor_as_float_vec(Tensor ws) {
  size_t size = reduce(ws.shape);
  if (ws.dtype == "float") {
    std::vector<float> src(reinterpret_cast<float *>(ws.data),
                           reinterpret_cast<float *>(ws.data) + size);
    return src;
  }
  throw std::runtime_error(
      "Other than float32[], format not supported. Received : " + ws.dtype);
}

static std::vector<uint8_t> get_tensor_as_uint8_t_vec(Tensor ws) {
  size_t size = reduce(ws.shape);
  if (ws.dtype == "uint8") {
    std::vector<uint8_t> src(reinterpret_cast<uint8_t *>(ws.data),
                             reinterpret_cast<uint8_t *>(ws.data) + size);
    std::vector<uint8_t> r(src.begin(), src.end());
    return r;
  }
  throw std::runtime_error(
      "other than uint8, format not supported Received : " + ws.dtype);
}

static std::vector<int32_t> get_tensor_as_int32_vec(Tensor ws) {
  size_t size = reduce(ws.shape);
  if (ws.dtype == "int32") {
    std::vector<int32_t> src(reinterpret_cast<int32_t *>(ws.data),
                             reinterpret_cast<int32_t *>(ws.data) + size);
    return src;
  }
  throw std::runtime_error(
      "Other than int32, format not supported. Received : " + ws.dtype);
}

template <typename T>
bool saveBufferToFile(const T *buffer, size_t numElements,
                      const std::string &filename) {
  // Open the file in output mode, truncating any existing content
  std::ofstream outFile(filename, std::ios::out | std::ios::trunc);

  // Check if file was opened successfully
  if (!outFile) {
    std::cerr << "Error: Unable to open file " << filename << " for writing."
              << std::endl;
    return false;
  }

  // Write each element to the file on a new line
  for (size_t i = 0; i < numElements; ++i) {
    outFile << buffer[i] << std::endl;
  }

  // Check if writing was successful
  if (outFile.fail()) {
    std::cerr << "Error: Failed to write to file " << filename << std::endl;
    outFile.close();
    return false;
  }

  // Close the file
  outFile.close();
  return true;
}

} // namespace OpsFusion
