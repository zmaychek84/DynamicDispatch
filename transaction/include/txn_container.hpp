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

#ifndef TRANSACTION_H
#define TRANSACTION_H

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

class Transaction {
public:
  static Transaction &getInstance() {
    static Transaction instance;
    return instance;
  }
  Transaction(Transaction const &) = delete;
  void operator=(Transaction const &) = delete;
  const std::string &get_txn_str(const std::string &);
  std::vector<std::uint8_t> get_txn_bvec(const std::string &);
  std::vector<std::string> match_prefix(const std::string &);
  Transaction();

  template <typename T>
  void GetBinData(const std::string &binaryData, std::vector<T> &outVector,
                  bool append = false) {
    const char *dataPtr = binaryData.data();
    size_t dataLen = binaryData.size();

    if (!append) {
      outVector.clear();
    }

    size_t currentSize = outVector.size();
    size_t newSize = dataLen / sizeof(T);
    size_t minSize = std::min(currentSize, newSize);

    for (size_t i = 0; i < minSize; ++i) {
      memcpy(&outVector[i], dataPtr + i * sizeof(T), sizeof(T));
    }

    for (size_t i = currentSize; i < newSize; ++i) {
      T value;
      memcpy(&value, dataPtr + i * sizeof(T), sizeof(T));
      outVector.push_back(value);
    }

    if (append && dataLen % sizeof(T) != 0) {
      T value = 0; // Zero-initialize the value
      memcpy(&value, dataPtr + newSize * sizeof(T), dataLen % sizeof(T));
      outVector.push_back(value);
    }
  }
};
#endif
