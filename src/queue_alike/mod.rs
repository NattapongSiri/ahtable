//! Subset operation of `VecDeque` specifically for the purpose of support `ArrayHashTable`.
//! To properly support operation such as `pop_front` on compound key where standard `VecDeque` require
//! drain or split_off to semantically move out the prefix from `VecDeque`.
//! 
//! In standard `VecDeque`, both `drain` and `split` will perform mem copy on pointed area 
//! to construct a value for that element which is in-efficient as we can re-use that memory area 
//! because in `ArrayHashTable`, the new data always `push_back` so `pop_front` is safe to assume that 
//! nobody will insert anything in front of it and cause two pointer to point to the same memory area. 
use std::alloc::*;
use core::mem::*;
use core::ptr;

/// Default size in bytes of this collection when construct without explicit size in number of element
const DEFAULT_INIT_SIZE: usize = 4096;

#[derive(Debug)]
pub(crate) struct QueueAlike<T> {
    /// Max number of element in this collection before it expand.
    capacity: usize,
    /// Current cursor which move when dequeued
    head: usize,
    /// Current  cursor which move when enqueue
    tail: usize,
    /// Current number of element in this of this collection, not number of bytes
    len: usize, 
    /// pointer to a place before first element 
    ptr: *mut T,
}

impl<T> Clone for QueueAlike<T> { 
    fn clone(&self) -> Self {
        unsafe {
            let ptr = alloc(Layout::array::<T>(self.capacity).unwrap()).cast();
            ptr::copy(self.ptr, ptr, self.len);
            QueueAlike {
                capacity: self.capacity,
                head: self.head,
                tail: self.tail,
                len: self.len,
                ptr
            }
        }
    }
}

impl<T> QueueAlike<T> {
    pub(crate) fn new() -> QueueAlike<T> {
        // let fragment = DEFAULT_INIT_SIZE % align_of::<T>(); // Check if default size is aligned
        let capacity = DEFAULT_INIT_SIZE / size_of::<T>();
        unsafe {
            let ptr = alloc(Layout::array::<T>(capacity).unwrap()).cast();
            QueueAlike {
                capacity,
                head: 0,
                tail: 0,
                len: 0,
                ptr
            }
        }
    }

    /// Allocate new collection that can store upto given size before reallocation happen.
    /// 
    /// # Parameter
    /// `size` - Initial size in number of element of this collection
    /// 
    /// # Return
    /// New empty [QueueAlike](struct.QueueAlike.html)
    pub(crate) fn with_capacity(size: usize) -> QueueAlike<T> {
        if size == 0 {
            panic!("Size must be at least 1");
        }
        // let capacity = size * size_of::<T>();
        unsafe {
            let ptr = alloc(Layout::array::<T>(size).unwrap()).cast();
            QueueAlike {
                capacity: size,
                head: 0,
                tail: 0,
                len: 0,
                ptr
            }
        }
    }

    /// Get a mutably raw pointer that point to data buffer of this collection.
    /// 
    /// # Return
    /// `*mut T` - Pointer to data buffer
    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get a raw pointer that point to data buffer of this collection.
    /// 
    /// # Return
    /// `*mut T` - Pointer to data buffer
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Make underlying buffer continguous. This is useful for a case where
    /// the collection contains multiple different type concatenated by intermediate
    /// other type which will be convert back to original type by using underlying raw
    /// pointer. In most case beside what mention above, you shouldn't need to call this.
    /// 
    /// An example use case is when user want to reference into original data within this collection 
    /// which represent by byte in this collection, it'd not be possible to use `core::mem::transmute`
    /// on non-contiguous pointer where half of it on the tail and another half is on the head.
    /// With contiguous pointer, such scenario will never happen.
    /// 
    /// It guarantee that after this method return, the underlying pointer will have first data
    /// at first pointee of pointer then subsequence value will follow this pointee.
    pub(crate) fn contiguous(&mut self) {
        unsafe {
            if self.head == 0 {
                // It's already contiguous
                return
            } else if self.tail < self.head {
                // The tail is in front of head which mean the data in buffer wrap around
                // the end of buffer. There'll be two chunk to copy`
                let layout = Layout::array::<T>(self.capacity).unwrap();
                let new_buffer = alloc(layout).cast();
                let first_chunk = self.capacity - self.head;
                ptr::copy(self.ptr, new_buffer, first_chunk);
                ptr::copy(self.ptr, new_buffer.add(first_chunk), self.tail);
                self.head = 0;
                self.tail = self.len;
                // We don't need to copy back the data. We simply swap the pointer and just
                // dealloc the old pointer.
                // We don't need to worry about memory leak as all of the data still has owner.
                dealloc(self.ptr.cast(), layout);
                self.ptr = new_buffer;
            } else if self.tail > self.head {
                // Tail is beyond head. It mean that there's some continguous data inside buffer
                // but the first value has some non-zero offset. Since caller call this method
                // need to expect some cost, we took this chance to normalize entire data so
                // the offset from head become zero. It'll be cheaper for iterator to iterate
                // as iterator doesn't need to have a modulo operation on each iteration.
                let layout = Layout::array::<T>(self.capacity).unwrap();
                let new_buffer = alloc(layout).cast();
                ptr::copy(self.ptr, new_buffer, self.len);
                self.head = 0;
                self.tail = self.len;
                dealloc(self.ptr.cast(), layout);
                self.ptr = new_buffer;
            } else {
                // No data. Simply reset head and tail pointer.
                self.tail = 0;
                self.head = 0;
            }
        }
    }

    /// Return true if the data within buffer is in sequence from left to right.
    /// In other word, tail pointer is behind head pointer.
    /// 
    /// It doesn't guarantee that the first pointee of pointer has data.
    #[inline]
    pub(crate) fn is_contiguous(&self) -> bool {
        self.tail >= self.head
    }

    /// Get first pointee in underlying pointer to buffer
    #[inline]
    pub(crate) fn head(&self) -> usize {
        self.head
    }

    /// Get last pointee in underlying pointer to buffer
    #[inline]
    pub(crate) fn tail(&self) -> usize {
        self.tail
    }

    /// Put data at tail of this collection. It may grow internal buffer by twice the old size if
    /// the buffer is full.
    pub(crate) fn enqueue(&mut self, value: T) {
        unsafe {
            if self.len == self.capacity {
                self.expand();
            }
            let index = self.tail % self.capacity;
            ptr::write(self.ptr.add(index), value);
            self.tail += 1;
            self.len += 1;
        }
    }

    /// Remove first element of this collection. 
    pub(crate) fn dequeue(&mut self) -> Option<T> {
        if self.len == 0 {
            // All elements are dequeued.

            // Took this chance to reset head and tail
            self.head = 0;
            self.tail = 0;
            return None
        }
        unsafe {
            let index = self.head % self.capacity;
            self.head += 1;
            self.len -= 1;
            Some(ptr::read(self.ptr.add(index)))
        }
    }


    /// Remove `n` elements of this collection and return it as `Vec<T>`.
    /// This method will be a bit faster than iterate over sub-queue to collect `Vec`
    /// because it directly copy pointer to internal buffer of this collection
    /// into `Vec` then update the cursor only once.
    /// 
    /// # Parameter
    /// `n` - Number of elements to be dequeue. If it's 1, it's equals to normal
    /// [dequeue](struct.QueueAlike.html#method.dequeue). If it's 0, it return `None`
    /// 
    /// # Return
    /// `Some(Vec<T>)` for `n >= 1 && n < self.len() && self.len > 0`. Otherwise, `None`
    pub(crate) fn dequeue_to(&mut self, n: usize) -> Option<Vec<T>> {
        if self.len == 0 {
            // All elements are dequeued.

            // Took this chance to reset head and tail
            self.head = 0;
            self.tail = 0;
            return None
        } else if n > self.len || n == 0 {
            return None
        }
        unsafe {
            let from = self.head % self.capacity;
            let to = (self.head + n) % self.capacity;
            let mut result = Vec::with_capacity(n);
            result.set_len(n);
            let buffer = result.as_mut_ptr();

            if to < from {
                let chunk_size = self.capacity - from;
                ptr::copy(self.ptr.add(from), buffer, chunk_size);
                ptr::copy(self.ptr, buffer.add(chunk_size), to);
            } else {
                ptr::copy(self.ptr.add(from), buffer, n);
            }
            self.head += n;
            self.len -= n;
            Some(result)
        }
    }

    /// Identify an index where the element starting at that position has exactly given
    /// sequence.
    /// For example, if this collection contains: [1, 2, 3, 4, 5, 3, 4] and caller give
    /// &[3, 4, 5] slice, it will return Some(2).
    /// 
    /// # Parameter
    /// `sequence` - A slice of sequence to look up in this collection
    /// 
    /// # Return
    /// `Option<usize>` where `Some(usize)` contains an index of the first element being matched
    /// with given sequence. If there's no match, it return `None`
    pub(crate) fn match_sequence(&self, sequence: &[T]) -> Option<usize> where T: PartialEq {
        let mut matched = 0;
        for i in 0..self.len {
            unsafe {
                let elm = self.ptr.add((self.head + i) % self.capacity);
                if *elm == sequence[matched] {
                    matched += 1;
                    if matched == sequence.len() {
                        return Some(i - matched)
                    }
                } else {
                    // Need to backtrack matched to find sub-matched pattern,
                    // for example, if collection has 1 1 1 1 2 3 and sequence is
                    // 1 1 2 3, after i >= 1, matched shall never goes down below 1
                    while matched > 0 {
                        matched -= 1;
                        if *elm == sequence[matched] {
                            break
                        }
                    }
                }
            }
        }
        None
    }

    /// Remove element at given index out of this collection.
    /// This operation have `O(2n)` for both time and memory requirement 
    /// where `n` is the number of element behind the element being remove.
    /// 
    /// # Parameter
    /// - `index` - An index of element to be removed from this queue
    /// 
    /// # Return
    /// An `Ok(T)` where `T` is element being removed. `Err(RemoveErr)` if the index is beyond last
    /// element of collection. This method will panic if there's insufficient memory available.
    pub fn remove(&mut self, index: usize) -> Result<T, RemoveErr> {
        if index > self.len {
            return Err(RemoveErr{})
        } else if index == self.len {
            if self.tail != 0 {
                self.tail -= 1;
            } else {
                self.tail = self.len;
            }
            self.len -= 1;
            unsafe {return Ok(ptr::read(self.ptr.add(self.tail % self.capacity)))};
        } else if index == 0 {
            let old_head = self.head % self.capacity;
            self.head += 1;
            self.len -= 1;
            unsafe {return Ok(ptr::read(self.ptr.add(old_head)))};
        }
        let first = (self.head + index + 1) % self.capacity;
        
        unsafe {
            let elm = ptr::read(self.ptr.add((self.head + index) % self.capacity));
            let layout = Layout::array::<T>(self.len() - index).unwrap();
            let buffer = alloc(layout).cast();
            if self.tail < self.head {
                let chunk_len = self.len - first;
                ptr::copy(self.ptr.add(first), buffer, chunk_len);
                ptr::copy(self.ptr, buffer.add(chunk_len), self.tail);
                ptr::copy(buffer, self.ptr.add((self.head + index) % self.capacity), chunk_len + 1);
                ptr::copy(buffer.add(chunk_len + 1), self.ptr, self.len - chunk_len + 1);
            } else {
                let len = self.tail - first;
                ptr::copy(self.ptr.add(first), buffer, len);
                ptr::copy(buffer, self.ptr.add((self.head + index) % self.capacity), len);
            };
            dealloc(buffer.cast(), layout);
            self.tail -= 1;
            self.len -= 1;
            Ok(elm)
        }
    }

    /// Remove a range of index from current collection and return it as `Vec<T>`.
    /// 
    /// It may cause the slot become non-contiguous in the future if the range to be remove
    /// is either unbound or 0.
    /// 
    /// # Parameter
    /// `range` - Any kind of range such as `0..2`, `..2`, `1..`, or `..=3`. The range must
    /// be 0 up to length of this collection.
    /// 
    /// # Return
    /// Ok(Vec<T>) if given range is within this collection.
    pub fn remove_within<R>(&mut self, range: R) -> Result<Vec<T>, RemoveErr> where R: core::ops::RangeBounds<usize> {
        use core::ops::Bound::*;
        let first = match range.start_bound() {
            Included(i) => self.head + *i,
            Excluded(i) => self.head + i + 1,
            Unbounded => self.head
        };
        let last = match range.end_bound() {
            Included(i) => self.head + *i + 1,
            Excluded(i) => self.head + *i,
            Unbounded => self.tail
        };
        let len = last - first;
        if len > self.len || len == 0 {
            return Err(RemoveErr {})
        } else if first == self.head {
            return self.dequeue_to(len).ok_or(RemoveErr {});
        }

        let mut result = Vec::with_capacity(len);
        let removing_tail = last % self.capacity;
        let removing_head = first % self.capacity;

        unsafe {
            result.set_len(len);
            let buffer = result.as_mut_ptr();
            if removing_tail > removing_head || removing_tail == 0 {
                // Tail ptr is beyond head so it's contiguous chunk
                ptr::copy(self.ptr.add(removing_head), buffer, len);
            } else {
                // Tail ptr is ahead of head so it's two incontiguous chunks.
                let first_chunk = self.capacity - removing_head;
                ptr::copy(self.ptr.add(removing_head), buffer, first_chunk);
                ptr::copy(self.ptr, buffer.add(first_chunk), removing_tail);
            }
            let remain_tail = self.tail % self.capacity;

            if remain_tail > removing_tail {
                // There's one contiguous tail to copy back to fill the void
                let fill_back_size = remain_tail - removing_tail;
                let layout = Layout::array::<T>(fill_back_size).unwrap();
                let buf = alloc(layout).cast();
                ptr::copy(self.ptr.add(removing_tail), buf, fill_back_size);
                Self::wrapping_fill_back(buf, self.ptr, fill_back_size, removing_head, self.capacity);
                dealloc(buf.cast(), layout);
            } else if remain_tail < removing_tail {
                // There's two chunks to copy into buffer to properly fill the void
                let fill_back_size = self.capacity - removing_tail + remain_tail;
                let layout = Layout::array::<T>(fill_back_size).unwrap();
                let buf = alloc(layout).cast();
                let first_chunk = self.capacity - removing_tail;
                ptr::copy(self.ptr.add(removing_tail), buf, first_chunk);
                ptr::copy(self.ptr, buf.add(first_chunk), remain_tail);
                Self::wrapping_fill_back(buf, self.ptr, fill_back_size, removing_head, self.capacity);
                dealloc(buf.cast(), layout);
            }
            // If remain_tail == removing_tail, there's nothing to copy

            self.tail -= len;
            self.len -= len;
        }

        Ok(result)
    }

    /// Similar to [remove_within](struct.QueueAlike.html#method.remove_within) but doesn't drop
    /// nor return the element being removed. This is unsafe because it won't call drop on element
    /// being removed. 
    /// 
    /// It is intended to be used in case where caller already use other unsafe method to take
    /// ownership of these elements. Calling [remove_within](struct.QueueAlike.html#method.remove_within)
    /// will cause two instances own the same data. It may lead to seg-fault.
    /// 
    /// It may cause the slot become non-contiguous in the future if the range to be remove
    /// is either unbound or 0.
    /// 
    /// # Parameter
    /// `range` - Any kind of range such as `0..2`, `..2`, `1..`, or `..=3`. The range must
    /// be 0 up to length of this collection.
    /// 
    /// # Return
    /// Ok(()) if given range is within this collection.
    pub(crate) unsafe fn silent_remove_within<R>(&mut self, range: R) -> Result<(), RemoveErr> where R: core::ops::RangeBounds<usize> {
        use core::ops::Bound::*;
        let first = match range.start_bound() {
            Included(i) => self.head + *i,
            Excluded(i) => self.head + i + 1,
            Unbounded => self.head
        };
        let last = match range.end_bound() {
            Included(i) => self.head + *i + 1,
            Excluded(i) => self.head + *i,
            Unbounded => self.tail
        };
        let len = last - first;
        if len > self.len || len == 0 {
            return Err(RemoveErr {})
        } else if first == self.head {
            self.head += len;
            self.len -= len;
            return Ok(())
        }

        let removing_tail = last % self.capacity;
        let removing_head = first % self.capacity;

        let remain_tail = self.tail % self.capacity;

        if remain_tail > removing_tail {
            // There's one contiguous tail to copy back to fill the void
            let fill_back_size = remain_tail - removing_tail;
            let layout = Layout::array::<T>(fill_back_size).unwrap();
            let buf = alloc(layout).cast();
            ptr::copy(self.ptr.add(removing_tail), buf, fill_back_size);
            Self::wrapping_fill_back(buf, self.ptr, fill_back_size, removing_head, self.capacity);
            dealloc(buf.cast(), layout);
        } else if remain_tail < removing_tail {
            // There's two chunks to copy into buffer to properly fill the void
            let fill_back_size = self.capacity - removing_tail + remain_tail;
            let layout = Layout::array::<T>(fill_back_size).unwrap();
            let buf = alloc(layout).cast();
            let first_chunk = self.capacity - removing_tail;
            ptr::copy(self.ptr.add(removing_tail), buf, first_chunk);
            ptr::copy(self.ptr, buf.add(first_chunk), remain_tail);
            Self::wrapping_fill_back(buf, self.ptr, fill_back_size, removing_head, self.capacity);
            dealloc(buf.cast(), layout);
        }
        // If remain_tail == removing_tail, there's nothing to copy

        self.tail -= len;
        self.len -= len;

        Ok(())
    }

    /// Remove an item out of this queue. If there's multiple element which is equals to this item,
    /// it will remove just one of it. There is no guarantee on which one will be removed. The
    /// removed item will be return to caller
    /// 
    /// # Parameter
    /// `other` - an item to use as comparator to check if an item match with it.
    /// 
    /// # Return
    /// `Some(T)` if there's a match. None if there is no match.
    pub(crate) fn remove_one<O>(&mut self, other: &O) -> Option<T> where T: PartialEq<O> {
        let mut remove_position = None;
        let head = self.head % self.capacity;
        unsafe {
            for i in head..(head + self.len) {
                if *self.ptr.add(i % self.capacity) == *other {
                    remove_position = Some(i);
                    break;
                }
            }

            if let Some(i) = remove_position {
                if i == head {
                    return self.dequeue()
                }
                let j = (i + 1) % self.capacity; // position to move
                let tail = self.tail % self.capacity;
                let result = Some(ptr::read(self.ptr.add(i)));
                
                // `j` must be greater than 0 otherwise tail can't be less than it
                if tail < j {
                    // two chunks to copy
                    let first_chunk = self.capacity - j;
                    let size = tail + first_chunk;
                    let layout = Layout::array::<T>(size).unwrap();
                    let buffer = alloc(layout).cast();
                    ptr::copy(self.ptr.add(j), buffer, first_chunk);
                    ptr::copy(self.ptr, buffer.add(first_chunk), tail);
                    Self::wrapping_fill_back(buffer, self.ptr, size, i, self.capacity);
                    dealloc(buffer.cast(), layout);
                } else if tail > j {
                    let size = tail - j;
                    let layout = Layout::array::<T>(size).unwrap();
                    let buffer = alloc(layout).cast();
                    ptr::copy(self.ptr.add(j), buffer, size);
                    Self::wrapping_fill_back(buffer, self.ptr, size, i, self.capacity);
                }
                // If `tail == j`, it has no trailing element so no need to copy

                self.tail -= 1;
                self.len -= 1;
                return result
            }
        }

        None
    }

    /// Split this collection by copy all value beyond given position into new collection
    /// then directly move tail cursor back to the position so next enqueue will overwrite
    /// the copied one. This method will leave current collection to have length equals to
    /// given position by simply modify the pointer. However, the underlying buffer is left
    /// untouched. The O(n) is constant O(c) where `c` equals to (self.len() - position).
    /// 
    /// It uses raw pointer to copy the value so `T` doesn't need to implement `Copy`.
    /// This shall not violate Rust borrowing rule because the original value become inaccessible
    /// and may got overwrite by future [enqueue](struct.QueueAlike.html#method.enqueue).
    /// 
    /// # Return
    /// Another [QueueAlike](struct.QueueAlike.html) that contains copied of other half.
    pub(crate) fn split_off(&mut self, position: usize) -> Result<QueueAlike<T>, SplitErr> {
        if position >= self.len {
            return Err(SplitErr {})
        }

        unsafe {
            let mut other = QueueAlike::with_capacity(self.len - position);
            let other_head = (self.head + position) % self.capacity;
            let other_tail = self.tail % self.capacity;
            if other_head >= other_tail {
                let mut chunk_len = self.capacity - other_head;
                ptr::copy(self.ptr.add(other_head), other.ptr, chunk_len);
                if other_tail > 0 {
                    ptr::copy(self.ptr, other.ptr.add(chunk_len), other_tail);
                    chunk_len += other_tail;
                }
                other.len = chunk_len;
                other.tail = chunk_len;
            } else {
                ptr::copy(self.ptr.add(other_head), other.ptr, other_tail - other_head);
                other.len = other_tail - other_head;
                other.tail = other.len;
            }

            self.len = position;
            // Set new tail position ignore the element that was copied
            self.tail = self.head + position; 

            Ok(other)
        }
    }

    /// Truncate this collection at given position. This method will simply move tail cursor
    /// back to given position without actually remove any data. However, with subsequence
    /// [enqueue](struct.QueueAlike.html#method.enqueue), it'll got overwritten.
    pub(crate) fn truncate(&mut self, position: usize) -> Result<(), TruncateErr> {
        if position > self.len {
            return Err(TruncateErr {});
        }
        self.len = position;
        self.tail = (self.head + position) % self.capacity;
        Ok(())
    }

    /// Borrow a sub-queue from given index to last element in this queue.
    /// # Parameter
    /// - `index` - An index of queue which will become first element of sub-queue
    /// 
    /// # Return
    /// It return [SubQueueAlike](struct.SubQueueAlike.html) which can be further
    /// make sub-queue or iterate over element inside of it.
    pub(crate) fn subset_from(&self, index: usize) -> SubQueueAlike<'_, T> {
        debug_assert!(index < self.len);
        SubQueueAlike {
            queue: self,
            start: (self.head + index) % self.capacity,
            len: self.len - index
        }
    }

    /// Borrow a sub-queue up to given index in this queue.
    /// # Parameter
    /// - `index` - An index of queue that will become last element of sub-queue.
    /// 
    /// # Return
    /// It return [SubQueueAlike](struct.SubQueueAlike.html) which can be further
    /// make sub-queue or iterate over element inside of it.
    pub(crate) fn subset_to(&self, index: usize) -> SubQueueAlike<'_, T> {
        debug_assert!(index <= self.len);
        SubQueueAlike {
            queue: self,
            start: self.head,
            len: index
        }
    }

    /// Borrow a sub-queue from given `from` index to given `to` index in this queue.
    /// 
    /// # Parameters
    /// - `from` - an index of queue which will become first element of sub-queue
    /// - `to` - an index of queue which will become last element of sub-queue
    /// 
    /// # Return
    /// It return [SubQueueAlike](struct.SubQueueAlike.html) which can be further
    /// make sub-queue or iterate over element inside of it.
    pub(crate) fn subset(&self, from: usize, to: usize) -> SubQueueAlike<'_, T> {
        debug_assert!(from < self.len);
        debug_assert!(to <= self.len);
        debug_assert!(from < to);
        SubQueueAlike {
            queue: self,
            start: (self.head + from) % self.capacity,
            len: to - from
        }
    }
    /// Mutably borrow a sub-queue from given index to last element in this queue.
    /// # Parameter
    /// - `index` - An index of queue which will become first element of sub-queue
    /// 
    /// # Return
    /// It return [SubQueueAlikeMut](struct.SubQueueAlikeMut.html) which can be further
    /// make sub-queue or iterate over element inside of it.
    pub fn subset_mut_from(&mut self, index: usize) -> SubQueueAlikeMut<'_, T> {
        debug_assert!(index < self.len);
        let start = (self.head + index) % self.capacity;
        let len = self.len - index;
        SubQueueAlikeMut {
            queue: self,
            start,
            len
        }
    }

    /// Mutably borrow a sub-queue up to given index in this queue.
    /// # Parameter
    /// - `index` - An index of queue that will become last element of sub-queue.
    /// 
    /// # Return
    /// It return [SubQueueAlikeMut](struct.SubQueueAlikeMut.html) which can be further
    /// make sub-queue or iterate over element inside of it.
    pub fn subset_mut_to(&mut self, index: usize) -> SubQueueAlikeMut<'_, T> {
        debug_assert!(index <= self.len);
        let start = self.head;
        SubQueueAlikeMut {
            queue: self,
            start,
            len: index
        }
    }

    /// Mutably borrow a sub-queue from given `from` index to given `to` index in this queue.
    /// 
    /// # Parameters
    /// - `from` - an index of queue which will become first element of sub-queue
    /// - `to` - an index of queue which will become last element of sub-queue
    /// 
    /// # Return
    /// It return [SubQueueAlikeMut](struct.SubQueueAlikeMut.html) which can be further
    /// make sub-queue or iterate over element inside of it.
    pub fn subset_mut(&mut self, from: usize, to: usize) -> SubQueueAlikeMut<'_, T> {
        debug_assert!(from < self.len);
        debug_assert!(to <= self.len);
        debug_assert!(from < to);
        let start = (self.head + from) % self.capacity;
        SubQueueAlikeMut {
            queue: self,
            start,
            len: to - from
        }
    }

    /// Get an iterator over this collection
    /// 
    /// # Return
    /// An iterator of type [QueueAlikeIter](struct.QueueAlikeIter.html)
    pub(crate) fn iter(&self) -> QueueAlikeIter<T> {
        QueueAlikeIter {
            queue: self,
            cursor: 0
        }
    }

    /// Get a mutable iterator over this collection
    /// 
    /// # Return
    /// A mutable iterator of type [QueueAlikeIterMut](struct.QueueAlikeIterMut.html)
    pub(crate) fn iter_mut(&mut self) -> QueueAlikeIterMut<T> {
        QueueAlikeIterMut {
            queue: self,
            cursor: 0
        }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
    /// Perform wrapping fillback from source into dest. If `dest_offset` + `dest_len` is smaller than 
    /// `source_len`, it will split source into two chunks. One will be concatenate to fit the remaining
    /// in `dest` then the remaining `source` will be written into front of pointer
    #[inline]
    unsafe fn wrapping_fill_back(source: *mut T, dest: *mut T, source_len: usize, dest_offset: usize, dest_len: usize) {
        debug_assert!(source_len <= dest_len);
        let remain = dest_len - dest_offset;
        if remain < source_len {
            // need to split into two chunks
            ptr::copy(source, dest.add(dest_offset), remain);
            ptr::copy(source.add(remain), dest, source_len - remain);
        } else {
            // can simply copy it in as space behind offset is sufficient
            ptr::copy(source, dest.add(dest_offset), source_len);
        }
    }

    /// Expand the buffer of this collection by 2 times the old size.
    unsafe fn expand(&mut self) {
        let capacity = 2 * self.capacity;
        // Allocate new buffer
        let layout = Layout::array::<T>(capacity).unwrap();
        let raw = alloc(layout);
        let new_ptr = raw.cast();
        let head = self.head % self.capacity;
        let tail = self.tail % self.capacity;
        if head >= tail { 
            // head == tail can be either this collection is emptied or fulled.
            // Since this is expand method, it shall never be called when it is emptied.
            // We therefore, copy from head to end of buffer first to reset it head/tail position
            // in new buffer into `0` and `len`
            let len = self.len - head;
            ptr::copy(self.ptr.add(head), new_ptr, len);
            if tail > 0 {
                // Copy all the rest 
                ptr::copy(self.ptr, new_ptr.add(len), tail);
            }
        } else {
            // Simple case, copy from head to tail
            ptr::copy(self.ptr.add(head), new_ptr, self.tail - head);
        }
        // Free old buffer
        let old_layout = Layout::array::<T>(self.capacity).unwrap();
        dealloc(self.ptr.cast(), old_layout);
        self.capacity = capacity;
        self.ptr = new_ptr;
        self.head = 0;
        self.tail = self.len;
    }
}

/// Move internal buffer of vec into this [QueueAlike](struct.QueueAlike.html) object
/// without copy it. This operation is cheap.
impl<T> From<Vec<T>> for QueueAlike<T> {
    fn from(mut v: Vec<T>) -> QueueAlike<T> {
        let queue = QueueAlike {
            ptr: v.as_mut_ptr(),
            head: 0,
            tail: v.len(),
            len: v.len(),
            capacity: v.capacity()
        };
        
        core::mem::forget(v);
        queue
    }
}

/// An iterator that return a reference to value inside [QueueAlike](struct.QueueAlike.html)
pub(crate) struct QueueAlikeIter<'a, T> {
    queue: &'a QueueAlike<T>,
    cursor: usize
}

impl<'a, T> Iterator for QueueAlikeIter<'a, T> {
    type Item=&'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len {
            let offset = (self.cursor + self.queue.head) % self.queue.capacity;
            self.cursor += 1;
            unsafe {
                return Some(&*self.queue.ptr.add(offset))
            }
        }
        None
    }
}

impl<'a, T> core::iter::FusedIterator for QueueAlikeIter<'a, T> {}

impl<'a, T> ExactSizeIterator for QueueAlikeIter<'a, T> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

/// An iterator that return a reference to value inside [QueueAlike](struct.QueueAlike.html)
pub(crate) struct QueueAlikeIntoIter<T> {
    queue: QueueAlike<T>,
    cursor: usize
}

impl<T> Iterator for QueueAlikeIntoIter<T> {
    type Item=T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len {
            let offset = (self.cursor + self.queue.head) % self.queue.capacity;
            self.cursor += 1;
            unsafe {
                return Some(ptr::read(self.queue.ptr.add(offset)))
            }
        }
        None
    }
}

impl<T> core::iter::FusedIterator for QueueAlikeIntoIter<T> {}

impl<T> ExactSizeIterator for QueueAlikeIntoIter<T> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

impl<T> core::ops::Index<usize> for QueueAlike<T> {
    type Output=T;

    fn index(&self, index: usize) -> &T {
        unsafe {
            assert!(index < self.len());
            &*self.ptr.add((self.head + index) % self.capacity)
        }
    }
}

impl<T> core::ops::IndexMut<usize> for QueueAlike<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        unsafe {
            assert!(index < self.len());
            &mut *self.ptr.add((self.head + index) % self.capacity)
        }
    }
}

/// An iterator that return a mutable reference to value inside [QueueAlike](struct.QueueAlike.html)
pub(crate) struct QueueAlikeIterMut<'a, T> {
    queue: &'a mut QueueAlike<T>,
    cursor: usize
}

impl<'a, T> Iterator for QueueAlikeIterMut<'a, T> {
    type Item=&'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len {
            let offset = (self.cursor + self.queue.head) % self.queue.capacity;
            self.cursor += 1;
            unsafe {
                return Some(&mut *self.queue.ptr.add(offset))
            }
        }
        None
    }
}

impl<'a, T> core::iter::FusedIterator for QueueAlikeIterMut<'a, T> {}

impl<'a, T> ExactSizeIterator for QueueAlikeIterMut<'a, T> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

impl<'a, T> IntoIterator for &'a QueueAlike<T> {
    type Item=&'a T;
    type IntoIter=QueueAlikeIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        QueueAlikeIter {
            cursor: 0,
            queue: self
        }
    }
}

impl<'a, T> IntoIterator for &'a mut QueueAlike<T> {
    type Item=&'a mut T;
    type IntoIter=QueueAlikeIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        QueueAlikeIterMut {
            cursor: 0,
            queue: self
        }
    }
}

impl<T> IntoIterator for QueueAlike<T> {
    type Item=T;
    type IntoIter=QueueAlikeIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        QueueAlikeIntoIter {
            cursor: 0,
            queue: self
        }
    }
}

/// Provide len method that should return maximum length when iterate over it.
pub(crate) trait Bounded {
    fn len(&self) -> usize;
}

/// Represent a part of QueueAlike in an immutable subset of it
pub(crate) struct SubQueueAlike<'a, T> {
    queue: &'a QueueAlike<T>,
    start: usize,
    len: usize
}

impl<'a, T> SubQueueAlike<'a, T> {
    pub fn subset_from(&self, index: usize) -> SubQueueAlike<'a, T> {
        debug_assert!(index < self.len);
        SubQueueAlike {
            queue: self.queue,
            start: self.start + index,
            len: self.len - index
        }
    }

    pub fn subset_to(&self, index: usize) -> SubQueueAlike<'a, T> {
        debug_assert!(index <= self.len);
        SubQueueAlike {
            queue: self.queue,
            start: self.start,
            len: index
        }
    }

    pub fn subset(&self, from: usize, to: usize) -> SubQueueAlike<'a, T> {
        debug_assert!(from < self.len);
        debug_assert!(to <= self.len);
        debug_assert!(from < to);
        SubQueueAlike {
            queue: self.queue,
            start: self.start + from,
            len: to - from
        }
    }

    pub fn iter(&self) -> SubQueueAlikeIter<SubQueueAlike<T>> {
        SubQueueAlikeIter {
            cursor: 0,
            queue: self
        }
    }
}

impl<'a, T> Bounded for SubQueueAlike<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> core::ops::Index<usize> for SubQueueAlike<'a, T> {
    type Output=T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.queue[self.start + index]
    }
}

/// Represent a part of QueueAlike in an immutable subset of it
pub(crate) struct SubQueueAlikeMut<'a, T> {
    queue: &'a mut QueueAlike<T>,
    start: usize,
    len: usize
}

impl<'a, T> SubQueueAlikeMut<'a, T> {
    pub fn subset_from(&self, index: usize) -> SubQueueAlike<'_, T> {
        debug_assert!(index < self.len);
        SubQueueAlike {
            queue: self.queue,
            start: self.start + index,
            len: self.len - index
        }
    }

    pub fn subset_to(&self, index: usize) -> SubQueueAlike<'_, T> {
        debug_assert!(index <= self.len);
        SubQueueAlike {
            queue: self.queue,
            start: self.start,
            len: index
        }
    }

    pub fn subset(&self, from: usize, to: usize) -> SubQueueAlike<'_, T> {
        debug_assert!(from < self.len);
        debug_assert!(to <= self.len);
        debug_assert!(from < to);
        SubQueueAlike {
            queue: self.queue,
            start: self.start + from,
            len: to - from
        }
    }
    pub fn subset_mut_from(&mut self, index: usize) -> SubQueueAlikeMut<'_, T> {
        debug_assert!(index < self.len);
        SubQueueAlikeMut {
            queue: self.queue,
            start: self.start + index,
            len: self.len - index
        }
    }

    pub fn subset_mut_to(&mut self, index: usize) -> SubQueueAlikeMut<'_, T> {
        debug_assert!(index <= self.len);
        SubQueueAlikeMut {
            queue: self.queue,
            start: self.start,
            len: index
        }
    }

    pub fn subset_mut(&mut self, from: usize, to: usize) -> SubQueueAlikeMut<'_, T> {
        debug_assert!(from < self.len);
        debug_assert!(to <= self.len);
        debug_assert!(from < to);
        SubQueueAlikeMut {
            queue: self.queue,
            start: self.start + from,
            len: to - from
        }
    }

    pub fn iter(&'a self) -> SubQueueAlikeIter<'a, Self> {
        SubQueueAlikeIter {
            cursor: 0,
            queue: self
        }
    }

    pub fn iter_mut(&'a mut self) -> SubQueueAlikeIterMut<'a, T> {
        SubQueueAlikeIterMut {
            cursor: 0,
            queue: self
        }
    }
}

impl<'a, T> Bounded for SubQueueAlikeMut<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> core::ops::Index<usize> for SubQueueAlikeMut<'a, T> {
    type Output=T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.queue[self.start + index]
    }
}

impl<'a, T> core::ops::IndexMut<usize> for SubQueueAlikeMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.queue[self.start + index]
    }
}

pub(crate) struct SubQueueAlikeIter<'a, S> where S: core::ops::Index<usize> {
    cursor: usize,
    queue: &'a S
}

impl<'a, S> Iterator for SubQueueAlikeIter<'a, S> where S: Bounded + core::ops::Index<usize>, S::Output: Sized {
    type Item=&'a S::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len() {
            let cur = self.cursor;
            self.cursor += 1;
            Some(&self.queue[cur])
        } else {
            None
        }
    }
}

impl<'a, S> core::iter::FusedIterator for SubQueueAlikeIter<'a, S> where S: Bounded + core::ops::Index<usize>, S::Output: Sized {}

impl<'a, S> ExactSizeIterator for SubQueueAlikeIter<'a, S> where S: Bounded + core::ops::Index<usize>, S::Output: Sized {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

pub(crate) struct SubQueueAlikeIntoIter<'a, T> {
    cursor: usize,
    queue: SubQueueAlike<'a, T>
}

impl<'a, T> Iterator for SubQueueAlikeIntoIter<'a, T> {
    type Item=&'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len() {
            let cur = self.cursor;
            self.cursor += 1;
            unsafe {Some(&*self.queue.queue.ptr.add((self.queue.start + cur) % self.queue.queue.capacity))}
        } else {
            None
        }
    }
}

impl<'a, T> core::iter::FusedIterator for SubQueueAlikeIntoIter<'a, T> {}

impl<'a, T> ExactSizeIterator for SubQueueAlikeIntoIter<'a, T> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

impl<'a, T> IntoIterator for &'a SubQueueAlike<'a, T> {
    type Item=&'a T;
    type IntoIter=SubQueueAlikeIter<'a, SubQueueAlike<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        SubQueueAlikeIter {
            cursor: 0,
            queue: self
        }
    }
}

impl<'a, T> IntoIterator for SubQueueAlike<'a, T> {
    type Item=&'a T;
    type IntoIter=SubQueueAlikeIntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SubQueueAlikeIntoIter {
            cursor: 0,
            queue: self
        }
    }
}

pub(crate) struct SubQueueAlikeMutIntoIter<'a, T> {
    cursor: usize,
    queue: SubQueueAlike<'a, T>
}

impl<'a, T> Iterator for SubQueueAlikeMutIntoIter<'a, T> {
    type Item=&'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len() {
            let cur = self.cursor;
            self.cursor += 1;
            unsafe {Some(&mut *self.queue.queue.ptr.add((self.queue.start + cur) % self.queue.queue.capacity))}
        } else {
            None
        }
    }
}

impl<'a, T> core::iter::FusedIterator for SubQueueAlikeMutIntoIter<'a, T> {}

impl<'a, T> ExactSizeIterator for SubQueueAlikeMutIntoIter<'a, T> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

impl<'a, T> IntoIterator for &'a mut SubQueueAlikeMut<'a, T> {
    type Item=&'a mut T;
    type IntoIter=SubQueueAlikeIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SubQueueAlikeIterMut {
            cursor: 0,
            queue: self
        }
    }
}

impl<'a, T> IntoIterator for SubQueueAlikeMut<'a, T> {
    type Item=&'a mut T;
    type IntoIter=SubQueueAlikeMutIntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SubQueueAlikeMutIntoIter {
            cursor: 0,
            queue: SubQueueAlike {
                len: self.len,
                start: self.start,
                queue: self.queue
            }
        }
    }
}

pub(crate) struct SubQueueAlikeIterMut<'a, T> {
    cursor: usize,
    queue: &'a mut SubQueueAlikeMut<'a, T>
}

impl<'a, T> Iterator for SubQueueAlikeIterMut<'a, T> {
    type Item=&'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.queue.len() {
            let cur = self.cursor;
            self.cursor += 1;
            // Rust doesn't allow using IndexMut to retreive borrow mut from queue and return it here
            unsafe {
                Some(&mut *self.queue.queue.ptr.add((self.queue.start + cur) % self.queue.queue.capacity))
            }
        } else {
            None
        }
    }
}

impl<'a, T> core::iter::FusedIterator for SubQueueAlikeIterMut<'a, T> {}

impl<'a, T> ExactSizeIterator for SubQueueAlikeIterMut<'a, T> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

impl<T> Drop for QueueAlike<T> {
    fn drop(&mut self) {
        unsafe {
            // Call drop on all type stored inside this collection before dropping the memory of this collection
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut::<T>(self.ptr.cast(), self.len));
            dealloc(self.ptr.cast(), Layout::array::<T>(self.capacity).unwrap());
        }
    }
}
pub(crate) struct SplitErr {}

impl core::fmt::Debug for SplitErr {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("Split position is larger than number of elements in the collection")
    }
}
pub(crate) struct TruncateErr {}

impl core::fmt::Debug for TruncateErr {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("Truncate position is larger than number of elements in the collection")
    }
}

pub(crate) struct RemoveErr {}

impl core::fmt::Debug for RemoveErr {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("Remove position error. Either it is larger than number of elements in the collection or the range has zero length")
    }
}

#[cfg(test)]
mod tests;