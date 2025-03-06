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

#ifndef __CONT_REUSE__
#define __CONT_REUSE__

#include <memory>
#include <utility>
#include <vector>

namespace OpsFusion {
namespace Pass {
namespace ContReuse {

/// MemorySlot class
class MemorySlot {
public:
  MemorySlot() = default;
  MemorySlot(int start, int end, bool is_free);

  bool is_free() const { return is_free_; }
  void set_free() { is_free_ = true; }
  void set_occupied() { is_free_ = false; }
  int size() const { return end_ - start_; }
  int start() const { return start_; }
  int end() const { return end_; }

  static MemorySlot merge(const MemorySlot &slot1, const MemorySlot &slot2);
  static std::pair<MemorySlot, MemorySlot> split(const MemorySlot &slot,
                                                 int split_point);

private:
  int start_ = 0;
  int end_ = 0;
  bool is_free_ = false;
};

struct Span {
  Span() = default;
  Span(int start, int end) : start(start), end(end) {}
  Span(const MemorySlot &slot) : start(slot.start()), end(slot.end()) {}
  int size() const { return end - start; }
  operator bool() const { return (start != end); }
  static Span intersection(const Span &span1, const Span &span2) {
    if (span1.start >= span2.end || span2.start >= span1.end) {
      return Span();
    }
    return Span(std::max(span1.start, span2.start),
                std::min(span1.end, span2.end));
  }
  int start = 0;
  int end = 0;
};

bool operator==(const Span &lhs, const Span &rhs);
bool operator==(const Span &lhs, const MemorySlot &rhs);
bool operator==(const MemorySlot &lhs, const Span &rhs);

std::ostream &operator<<(std::ostream &os, const Span &span);

// Container interface
template <typename T> class IContainer {
public:
  virtual ~IContainer() = default;
  virtual bool empty() const = 0;
  virtual void insert(int index, const T &value) = 0;
  virtual void remove(int index) = 0;
  virtual T &at(int index) = 0;
  virtual const T &at(int index) const = 0;
  virtual int size() const = 0;
};

class IMemoryView {
public:
  virtual ~IMemoryView() = default;
  virtual void create_free_slot(size_t size) = 0;
  virtual Span get_free_slot(size_t size) = 0;
  virtual void return_free_slot(const Span &slot) = 0;
  virtual size_t size() const = 0;
  virtual size_t num_slots() const = 0;
};

class IBufferReuseAllocator {
public:
  virtual ~IBufferReuseAllocator() = default;
  virtual Span allocate(size_t size) = 0;
  virtual void deallocate(const Span &slot) = 0;
};

template <typename T> class VectorContainer : public IContainer<T> {
public:
  VectorContainer() = default;
  ~VectorContainer() = default;
  bool empty() const override;
  void insert(int index, const T &value) override;
  void remove(int index) override;
  T &at(int index) override;
  const T &at(int index) const override;
  int size() const override;

private:
  std::vector<T> data_;
};

class ISearchStrategy {
public:
  virtual ~ISearchStrategy() = default;
  virtual int search_free_slot(const IContainer<MemorySlot> &container,
                               size_t size) const = 0;
};

class FirstFit : public ISearchStrategy {
public:
  int search_free_slot(const IContainer<MemorySlot> &container,
                       size_t size) const override;
};

class BestFit : public ISearchStrategy {
public:
  int search_free_slot(const IContainer<MemorySlot> &container,
                       size_t size) const override;
};

class MemoryView : public IMemoryView {
public:
  MemoryView();
  MemoryView(std::unique_ptr<IContainer<MemorySlot>> slots,
             std::unique_ptr<ISearchStrategy> strategy);
  void create_free_slot(size_t size) override;
  Span get_free_slot(size_t size) override;
  void return_free_slot(const Span &slot) override;
  size_t size() const override;
  size_t num_slots() const override;

private:
  int search_free_slot(size_t size) const;
  void split_free_slot(int index, size_t size);
  void merge_free_slots(int index);
  int find_slot(const Span &slot) const;

private:
  std::unique_ptr<IContainer<MemorySlot>> slots_;
  std::unique_ptr<ISearchStrategy> strategy_;
};

class BufferReuseAllocator : public IBufferReuseAllocator {
public:
  BufferReuseAllocator(std::unique_ptr<IMemoryView> view);
  Span allocate(size_t size) override;
  void deallocate(const Span &slot) override;
  size_t size() const;

private:
  std::unique_ptr<IMemoryView> view_;
};

} // namespace ContReuse
} // namespace Pass
} // namespace OpsFusion

#endif // __CONT_REUSE__
