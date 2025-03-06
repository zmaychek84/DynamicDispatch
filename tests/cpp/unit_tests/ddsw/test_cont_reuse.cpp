
#include <gtest/gtest.h>
#include <memory>

#include "cont_reuse.hpp"

using namespace OpsFusion::Pass::ContReuse;

TEST(DDSW_ContReuse, SpanTest) {
  Span span1(0, 10);
  Span span2(5, 15);
  Span span3(10, 20);
  Span span4(15, 25);

  Span intersection1 = Span::intersection(span1, span2);
  EXPECT_EQ(intersection1, Span(5, 10));

  Span intersection2 = Span::intersection(span1, span3);
  EXPECT_EQ(intersection2, Span());

  Span intersection3 = Span::intersection(span1, span4);
  EXPECT_EQ(intersection3, Span());
}

TEST(DDSW_ContReuse, MemorySlotTest_Merge) {
  MemorySlot slot1(0, 10, true);
  MemorySlot slot2(10, 20, true);
  MemorySlot slot3(0, 10, false);
  MemorySlot slot4(10, 20, false);

  MemorySlot merged1 = MemorySlot::merge(slot1, slot2);
  MemorySlot merged2 = MemorySlot::merge(slot3, slot4);

  EXPECT_EQ(merged1.start(), 0);
  EXPECT_EQ(merged1.end(), 20);
  EXPECT_TRUE(merged1.is_free());

  EXPECT_EQ(merged2.start(), 0);
  EXPECT_EQ(merged2.end(), 20);
  EXPECT_FALSE(merged2.is_free());
}

TEST(DDSW_ContReuse, MemorySlotTest_Split) {
  MemorySlot slot(0, 10, true);

  auto split = MemorySlot::split(slot, 5);

  EXPECT_EQ(split.first.start(), 0);
  EXPECT_EQ(split.first.end(), 5);
  EXPECT_TRUE(split.first.is_free());

  EXPECT_EQ(split.second.start(), 5);
  EXPECT_EQ(split.second.end(), 10);
  EXPECT_TRUE(split.second.is_free());
}

TEST(DDSW_ContReuse, VectorContainerTest_Insert) {
  VectorContainer<int> container;
  container.insert(0, 1);
  container.insert(1, 2);
  container.insert(2, 3);

  EXPECT_EQ(container.size(), 3);
  EXPECT_EQ(container.at(0), 1);
  EXPECT_EQ(container.at(1), 2);
  EXPECT_EQ(container.at(2), 3);
}

TEST(DDSW_ContReuse, VectorContainerTest_Remove) {
  VectorContainer<int> container;
  container.insert(0, 1);
  container.insert(1, 2);
  container.insert(2, 3);

  container.remove(1);

  EXPECT_EQ(container.size(), 2);
  EXPECT_EQ(container.at(0), 1);
  EXPECT_EQ(container.at(1), 3);
}

TEST(DDSW_ContReuse, MemoryViewTest_createfreeslot) {
  MemoryView view;
  view.create_free_slot(10);
  view.create_free_slot(10);
  view.create_free_slot(20);

  EXPECT_EQ(view.size(), 40);
  EXPECT_EQ(view.num_slots(), 1);

  Span span1 = view.get_free_slot(5);
  EXPECT_EQ(span1, Span(0, 5));
  EXPECT_EQ(view.num_slots(), 2);

  Span span2 = view.get_free_slot(5);
  EXPECT_EQ(span2, Span(5, 10));
  EXPECT_EQ(view.num_slots(), 3);

  Span span3 = view.get_free_slot(5);
  EXPECT_EQ(span3, Span(10, 15));
  EXPECT_EQ(view.num_slots(), 4);

  view.return_free_slot(span1);
  EXPECT_EQ(view.num_slots(), 4);

  view.return_free_slot(span2);
  EXPECT_EQ(view.num_slots(), 3);

  view.return_free_slot(span3);
  EXPECT_EQ(view.num_slots(), 1);

  Span span4 = view.get_free_slot(20);
  EXPECT_EQ(span4, Span(0, 20));
  EXPECT_EQ(view.num_slots(), 2);

  Span span5 = view.get_free_slot(20);
  EXPECT_EQ(span5, Span(20, 40));
  EXPECT_EQ(view.num_slots(), 2);

  view.return_free_slot(span5);
  EXPECT_EQ(view.num_slots(), 2);

  view.return_free_slot(span4);
  EXPECT_EQ(view.num_slots(), 1);
}

TEST(DDSW_ContReuse, BufferReuseAllocator_all) {
  auto view = std::make_unique<MemoryView>();
  BufferReuseAllocator allocator(std::move(view));

  Span span1 = allocator.allocate(10);
  EXPECT_EQ(span1, Span(0, 10));

  Span span2 = allocator.allocate(10);
  EXPECT_EQ(span2, Span(10, 20));

  Span span3 = allocator.allocate(20);
  EXPECT_EQ(span3, Span(20, 40));
  EXPECT_EQ(allocator.size(), 40);

  allocator.deallocate(span1);
  allocator.deallocate(span2);
  allocator.deallocate(span3);
  EXPECT_EQ(allocator.size(), 40);
}
