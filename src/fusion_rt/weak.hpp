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

#pragma once
#include <assert.h>
#include <iostream>
#include <memory>
#include <unordered_map>
namespace vitis {
namespace ai {
template <typename T> struct WeakSingleton {
  static std::weak_ptr<T> the_instance_;
  template <typename... Args> static std::shared_ptr<T> create(Args &&...args) {
    std::shared_ptr<T> ret;
    if (the_instance_.expired()) {
      ret = std::make_shared<T>(std::forward<Args>(args)...);
      the_instance_ = ret;
    }
    ret = the_instance_.lock();
    assert(ret != nullptr);
    return ret;
  }
};
template <typename T> std::weak_ptr<T> WeakSingleton<T>::the_instance_;

// we don't support c++17 yet.
template <class...> using my_void_t = void;
template <typename T, class = void> struct invoke_initialize_if_possible {
  static void initialize(T *t) {}
};
template <typename T> struct WithInjection;
template <typename T>
using is_derived_from_with_injection =
    typename std::enable_if<std::is_base_of<WithInjection<T>, T>::value>::type;

template <typename T>
using is_not_derived_from_with_injection =
    typename std::enable_if<!std::is_base_of<WithInjection<T>, T>::value>::type;

template <typename T>
struct invoke_initialize_if_possible<T, is_derived_from_with_injection<T>> {
  static void initialize(T *t) {
    // with_injection<T>::create(...) invokes initialize() already,
    // void invoke it twice;
  }
};

template <typename T>
struct invoke_initialize_if_possible<
    T, my_void_t<decltype(std::declval<T>().initialize()),
                 // see comment above, otherwise, ambigurous template defined.
                 is_not_derived_from_with_injection<T>>> {
  static void initialize(T *t) { t->initialize(); }
};

template <typename K, typename T> struct WeakStore {
  static std::unordered_map<K, std::weak_ptr<T>> the_store_;
  template <typename... Args>
  static std::shared_ptr<T> create(const K &key, Args &&...args) {
    std::shared_ptr<T> ret;
    if (the_store_[key].expired()) {
      ret = create_1(std::forward<Args>(args)...);
      invoke_initialize_if_possible<T>::initialize(ret.get());
      the_store_[key] = ret;
    }
    ret = the_store_[key].lock();
    assert(ret != nullptr);
    return ret;
  }

  static std::shared_ptr<T> get(const K &key) {
    std::shared_ptr<T> ret;
    auto iter = the_store_.find(key);
    if (iter == the_store_.end()) {
      throw std::runtime_error("weak store key not found");
    }
    if (iter->second.expired()) { //  &&
      throw std::runtime_error("weak store key expired");
    }

    ret = the_store_[key].lock();
    assert(ret != nullptr);
    return ret;
  }

private:
  template <typename... Args>
  static typename std::enable_if<!std::is_constructible<T, Args...>::value,
                                 std::shared_ptr<T>>::type
  create_1(Args &&...args) {
    return T::create(std::forward<Args>(args)...);
  }
  template <typename... Args>
  static typename std::enable_if<std::is_constructible<T, Args...>::value,
                                 std::shared_ptr<T>>::type
  create_1(Args &&...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }
};
template <typename K, typename T>
std::unordered_map<K, std::weak_ptr<T>> WeakStore<K, T>::the_store_;

} // namespace ai
} // namespace vitis
