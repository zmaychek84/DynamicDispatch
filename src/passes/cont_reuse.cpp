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

#include "cont_reuse.hpp"
#include <iostream>
#include <stdexcept>

namespace OpsFusion {
namespace Pass {
namespace ContReuse {

MemorySlot::MemorySlot(int start, int end, bool is_free)
    : start_(start), end_(end), is_free_(is_free) {}

MemorySlot MemorySlot::merge(const MemorySlot &slot1, const MemorySlot &slot2) {
  if (slot1.end() != slot2.start()) {
    throw std::runtime_error("Cannot merge non-adjacent slots");
  }

  return MemorySlot(slot1.start(), slot2.end(),
                    slot1.is_free() && slot2.is_free());
}

std::pair<MemorySlot, MemorySlot> MemorySlot::split(const MemorySlot &slot,
                                                    int split_point) {
  if (split_point <= slot.start() || split_point >= slot.end()) {
    throw std::runtime_error("Cannot split slot at invalid point");
  }

  return std::make_pair(MemorySlot(slot.start(), split_point, slot.is_free()),
                        MemorySlot(split_point, slot.end(), slot.is_free()));
}

template <typename T> bool VectorContainer<T>::empty() const {
  return data_.empty();
}

template <typename T>
void VectorContainer<T>::insert(int index, const T &value) {
  if (index < 0 || index > data_.size()) {
    throw std::out_of_range("Invalid index");
  }

  data_.insert(data_.begin() + index, value);
}

template <typename T> void VectorContainer<T>::remove(int index) {
  if (index < 0 || index >= data_.size()) {
    throw std::out_of_range("Invalid index");
  }

  data_.erase(data_.begin() + index);
}

template <typename T> T &VectorContainer<T>::at(int index) {
  if (index < 0 || index >= data_.size()) {
    throw std::out_of_range("Invalid index");
  }

  return data_[index];
}

template <typename T> const T &VectorContainer<T>::at(int index) const {
  if (index < 0 || index >= data_.size()) {
    throw std::out_of_range("Invalid index");
  }

  return data_[index];
}

template <typename T> int VectorContainer<T>::size() const {
  return static_cast<int>(data_.size());
}

template class VectorContainer<int>;
template class VectorContainer<MemorySlot>;

int FirstFit::search_free_slot(const IContainer<MemorySlot> &container,
                               size_t size) const {
  for (int i = 0; i < container.size(); ++i) {
    if (container.at(i).is_free() && container.at(i).size() >= size) {
      return i;
    }
  }

  return -1;
}

MemoryView::MemoryView()
    : slots_(std::make_unique<VectorContainer<MemorySlot>>()),
      strategy_(std::make_unique<FirstFit>()) {}

MemoryView::MemoryView(std::unique_ptr<IContainer<MemorySlot>> slots,
                       std::unique_ptr<ISearchStrategy> strategy)
    : slots_(std::move(slots)), strategy_(std::move(strategy)) {}

void MemoryView::create_free_slot(size_t size_) {
  auto size = static_cast<int>(size_);
  int start = slots_->empty() ? 0 : slots_->at(slots_->size() - 1).end();
  slots_->insert(slots_->size(), MemorySlot(start, start + size, true));
  merge_free_slots(slots_->size() - 1);
}

Span MemoryView::get_free_slot(size_t size_) {
  auto size = static_cast<int>(size_);
  int index = strategy_->search_free_slot(*slots_, size);
  if (index == -1) {
    create_free_slot(size);
    index = slots_->size() - 1;
  }

  MemorySlot &slot = slots_->at(index);
  if (slot.size() > size) {
    auto split = MemorySlot::split(slot, slot.start() + size);
    slots_->insert(index + 1, split.second);
    slots_->at(index) = split.first;
  }
  slots_->at(index).set_occupied();
  return Span(slots_->at(index));
}

void MemoryView::return_free_slot(const Span &slot) {
  // find the slot in the container
  int index = find_slot(slot);
  if (slots_->at(index).is_free()) {
    throw std::runtime_error("Slot is already free");
  }
  slots_->at(index).set_free();
  merge_free_slots(index);
}

size_t MemoryView::size() const {
  if (slots_->empty()) {
    return 0;
  }

  return slots_->at(slots_->size() - 1).end();
}

size_t MemoryView::num_slots() const { return slots_->size(); }

int MemoryView::search_free_slot(size_t size) const {
  return strategy_->search_free_slot(*slots_, size);
}

void MemoryView::split_free_slot(int index, size_t size_) {
  auto size = static_cast<int>(size_);
  auto split =
      MemorySlot::split(slots_->at(index), slots_->at(index).start() + size);
  slots_->at(index) = split.first;
  slots_->insert(index + 1, split.second);
}

void MemoryView::merge_free_slots(int index) {
  // if slot at index is not the last slot, merge it with next slot
  if ((index < slots_->size() - 1) && (slots_->at(index + 1).is_free())) {
    MemorySlot merged =
        MemorySlot::merge(slots_->at(index), slots_->at(index + 1));
    slots_->remove(index + 1);
    slots_->at(index) = merged;
  }

  // if slot at index is not the first slot, merge it with previous slot
  if ((index > 0) && (slots_->at(index - 1).is_free())) {
    MemorySlot merged =
        MemorySlot::merge(slots_->at(index - 1), slots_->at(index));
    slots_->remove(index);
    slots_->at(index - 1) = merged;
  }
}

int MemoryView::find_slot(const Span &slot) const {
  for (int i = 0; i < slots_->size(); ++i) {
    if (slots_->at(i) == slot) {
      return i;
    }
  }

  throw std::runtime_error("Slot not found");
}

bool operator==(const Span &lhs, const Span &rhs) {
  return lhs.start == rhs.start && lhs.end == rhs.end;
}

bool operator==(const Span &lhs, const MemorySlot &rhs) {
  return lhs.start == rhs.start() && lhs.end == rhs.end();
}

bool operator==(const MemorySlot &lhs, const Span &rhs) {
  return lhs.start() == rhs.start && lhs.end() == rhs.end;
}

std::ostream &operator<<(std::ostream &os, const Span &span) {
  os << "[" << span.start << ", " << span.end << "]";
  return os;
}

int BestFit::search_free_slot(const IContainer<MemorySlot> &container,
                              size_t size) const {
  return -1;
}

BufferReuseAllocator::BufferReuseAllocator(std::unique_ptr<IMemoryView> view)
    : view_(std::move(view)) {}

Span BufferReuseAllocator::allocate(size_t size) {
  auto span = view_->get_free_slot(size);
  if (!span) {
    throw std::runtime_error("Failed to allocate memory");
  }
  return span;
}

void BufferReuseAllocator::deallocate(const Span &slot) {
  view_->return_free_slot(slot);
}

size_t BufferReuseAllocator::size() const { return view_->size(); }

} // namespace ContReuse
} // namespace Pass
} // namespace OpsFusion
