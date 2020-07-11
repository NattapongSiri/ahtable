//! An implementation of ArrayHash
//! 
//! ArrayHash is a data structure where the index is determined by hash and
//! each entry in array is a `Vec` that store data and all it collision.
//! 
//! Oritinal paper can be found [here](Askitis, N. & Zobel, J. (2005), Cache-conscious collision resolution for string hash tables, in ‘Proc. SPIRE String Processing and Information Retrieval Symp.’, Springer-Verlag, pp. 92–104)
//! 
//! This implementation try to use generic wherever possible.
//! It end up with ArrayHash that take anything that is clonable as value and anything that
//! implement `Hash`, `PartialEq`, and `Clone` as key.
//! It let you choose whichever `Hasher` that you want. The only constraint is that
//! `Hasher` must implement `Clone` trait.
//! 
//! It supports read only iteration, mutably iteration, and owned iteration.
//! 
//! To create [ArrayHash](struct.ArrayHash.html) use [ArrayHashBuilder](struct.ArrayHashBuilder.html).
//! The default `Hasher` is `XxHasher64`.

use twox_hash::{RandomXxHashBuilder64, XxHash64};
use core::hash::BuildHasher;

use queue_alike::{QueueAlike};

mod queue_alike;

const MAX_LOAD_FACTOR: usize = 100_000; // Number of element before resize the table

// Make each bucket fit into single memory page
const DEFAULT_BUCKETS_SIZE: usize = 4096 / std::mem::size_of::<usize>(); 
const DEFAULT_SLOT_SIZE: usize = 8;

/// A builder that use for build an [ArrayHash](struct.ArrayHash.html).
pub struct ArrayHashBuilder<H> {
    hasher: H,
    buckets_size: usize,
    max_load_factor: usize,
    slot_size: usize
}

/// Create new ArrayHashBuilder with default hasher and size.
/// As currently is, the default allocated number of slot per bucket is (4096 / size of usize) slots.
/// Each slot has 8 elements. It will use `XxHash64` as default hasher
/// 
/// Since all slots are Vec, it will be re-allocate if it grow larger than this default.
/// The number of slots per bucket will be held until the number of entry grew pass `max_load_factor`.
/// When it reach the `max_load_factor`, it will double the bucket size.
impl Default for ArrayHashBuilder<XxHash64> {
    fn default() -> ArrayHashBuilder<XxHash64> {
        ArrayHashBuilder {
            hasher: RandomXxHashBuilder64::default().build_hasher(),
            buckets_size: DEFAULT_BUCKETS_SIZE,
            max_load_factor: MAX_LOAD_FACTOR,
            slot_size: DEFAULT_SLOT_SIZE
        }
    }
}

impl<H> ArrayHashBuilder<H> where H: core::hash::Hasher {
    /// Create new ArrayHashBuilder by using given hasher.
    /// As currently is, the default allocated number of slot per bucket is (4096 / size of usize) slots.
    /// 
    /// Since all slots are Vec, it will be re-allocate if it grow larger than this default.
    /// However, number of slots per bucket will be constant. It will never grow pass this number.
    pub fn with_hasher(hasher: H) -> ArrayHashBuilder<H> {
        ArrayHashBuilder {
            hasher: hasher,
            buckets_size: DEFAULT_BUCKETS_SIZE,
            max_load_factor: MAX_LOAD_FACTOR,
            slot_size: DEFAULT_SLOT_SIZE
        }
    }

    /// Switch hasher to other hasher. This will consume current builder and
    /// return a new one with new builder
    pub fn hasher<H2>(self, hasher: H2) -> ArrayHashBuilder<H2> {
        ArrayHashBuilder {
            hasher,
            buckets_size: self.buckets_size,
            max_load_factor: self.max_load_factor,
            slot_size: self.slot_size
        }
    }

    /// Change buckets size of [ArrayHasher](struct.ArrayHasher.html).
    /// Buckets size will remain constant until number of item added surpass
    /// max_load_factor then it will be doubled.
    pub fn buckets_size(mut self, size: usize) -> Self {
        self.buckets_size = size;
        self
    }

    /// Change max number of entry before double the buckets size.
    /// After it is reached, the factor itself will be double.
    /// 
    /// For example, if original max_load_factor is 2, after
    /// two entries are added, the buckets_size and max_load_factor
    /// will be double. So max_load_factor become 4.
    pub fn max_load_factor(mut self, factor: usize) -> Self {
        self.max_load_factor = factor;
        self
    }

    /// Default initialized slot size in bytes. Every slot in bucket will be 
    /// allocated by given size. 
    /// Keep in mind that each slot is a `vec`. It can grow pass this
    /// number. It'd be best to give a proper estimation to prevent unnecessary
    /// re-allocation.
    pub fn slot_size(mut self, size: usize) -> Self {
        self.slot_size = size;
        self
    }

    /// Consume this builder and construct a new [ArrayHash](struct.ArrayHash.html)
    pub fn build<K, V>(self) -> ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
        ArrayHash {
            buckets: Some(vec![ArraySlot::with_capacity(self.slot_size); self.buckets_size].into_boxed_slice()),
            hasher: self.hasher,
            capacity: self.buckets_size,
            max_load_factor: self.max_load_factor,
            size: 0
        }
    }
}

/// An Array based slot which flatten inner type by using unsafe Rust.
/// This implementation reflect closely to what original paper make with some minor change
/// such as the prefix is 4 bytes instead of variable 1 or 2 bytes.
#[derive(Clone, Debug)]
pub struct ArraySlot<K, V> {
    _k: core::marker::PhantomData<K>,
    _v: core::marker::PhantomData<V>,
    data: QueueAlike<u8>,
    value_bound: usize
}

impl<K, V> ArraySlot<K, V> {
    /// Create new empty ArraySlot
    /// 
    /// # Return
    /// An empty instance of [ArraySlot](struct.ArraySlot.html)
    pub fn new() -> ArraySlot<K, V> {
        ArraySlot {
            _k: core::marker::PhantomData,
            _v: core::marker::PhantomData,
            data: QueueAlike::new(),
            value_bound: core::mem::size_of::<V>()
        }
    }
    /// Pre-allocate underlying bytes storage by given size in bytes.
    /// 
    /// # Parameter
    /// `size` - A size in bytes of initial inner buffer. The inner buffer is `Vec` so
    /// it will expand when it full with the cost of moving all existing data to new location.
    /// 
    /// # Return
    /// An empty instance of [ArraySlot](struct.ArraySlot.html) with pre-allocated size in bytes.
    pub fn with_capacity(size: usize) -> ArraySlot<K, V> {
        ArraySlot {
            _k: core::marker::PhantomData,
            _v: core::marker::PhantomData,
            data: QueueAlike::with_capacity(size),
            value_bound: core::mem::size_of::<V>()
        }
    }

    /// Remove all elements in this collection and properly drop each key and value
    /// for each element.
    #[inline]
    pub fn clear(&mut self) {
        let mut cursor = self.data.head() % self.data.capacity();
        let tail = self.data.tail() % self.data.capacity();
        let record_ptr = self.data.as_mut_ptr();
        unsafe {
            while cursor != tail {
                let key_len = *record_ptr.cast::<usize>();
                let key_begin = cursor + core::mem::size_of::<usize>();
                let key_end = key_begin + key_len * core::mem::size_of::<K>();

                if key_end < self.data.capacity() {
                    // keys are in contiguous mem
                    core::ptr::drop_in_place(
                        core::ptr::slice_from_raw_parts_mut(record_ptr.add(key_begin).cast::<K>(), key_len)
                    );
                } else {
                    // Keys are in incontiguous mem. There can be two possible case,
                    // 1. Each key is contiguous
                    // 2. One of a key is non-contiguous
                    let key_fragment = (self.data.capacity() - key_begin) % core::mem::size_of::<K>();

                    if key_fragment != 0 {
                        // One of a key is non-contiguous
                        let key_remain = core::mem::size_of::<K>() - key_fragment;
                        core::ptr::drop_in_place(
                            core::ptr::slice_from_raw_parts_mut(record_ptr.add(key_begin).cast::<K>(), (self.data.capacity() - key_begin) / core::mem::size_of::<K>())
                        );
                        core::ptr::drop_in_place(
                            core::ptr::slice_from_raw_parts_mut(record_ptr.add(key_remain).cast::<K>(), (key_end - self.data.capacity()) / core::mem::size_of::<K>() + key_remain)
                        );
                        let layout = std::alloc::Layout::from_size_align(core::mem::size_of::<K>(), core::mem::align_of::<K>()).unwrap();
                        let temp = std::alloc::alloc(layout);
                        core::ptr::copy(record_ptr.add(self.data.capacity() - key_fragment), temp, key_fragment);
                        core::ptr::copy(record_ptr, temp.add(key_fragment), key_remain);
                        core::ptr::read(temp); // Read and drop this key fragment immediately
                        std::alloc::dealloc(temp, layout);
                    } else {
                        // Each key is contiguous. We can drop these keys as two slices.
                        core::ptr::drop_in_place(
                            core::ptr::slice_from_raw_parts_mut(record_ptr.add(key_begin).cast::<K>(), (self.data.capacity() - key_begin) / core::mem::size_of::<K>())
                        );
                        core::ptr::drop_in_place(
                            core::ptr::slice_from_raw_parts_mut(record_ptr.cast::<K>(), (key_end - self.data.capacity()) / core::mem::size_of::<K>())
                        );
                    }
                }

                let value_offset = key_end % self.data.capacity();
                let value_end = (value_offset + core::mem::size_of::<V>()) % self.data.capacity();

                if value_offset < value_end {
                    // value is in contiguous mem
                    core::ptr::read(record_ptr.add(value_offset).cast::<V>());
                } else {
                    // value is fragmented
                    let value_fragment = self.data.capacity() - value_offset;
                    let layout = std::alloc::Layout::from_size_align(core::mem::size_of::<V>(), core::mem::align_of::<V>()).unwrap();
                    let temp = std::alloc::alloc(layout);
                    core::ptr::copy(record_ptr.add(value_offset), temp, value_fragment);
                    core::ptr::copy(record_ptr, temp.add(value_fragment), value_end);
                    core::ptr::read(temp); // Read and drop this key fragment immediately
                    std::alloc::dealloc(temp, layout);
                }

                cursor = value_end;
            }
        }
    }

    /// Push key and value into this [ArraySlot](struct.ArraySlot.html).
    /// The value will be append to the back of this slot.
    /// 
    /// This method use unsafe transmute to turn key and value into bytes pointer then
    /// iterate on each bytes until all the bytes are added into this slot.
    /// 
    /// Since this slot is use exclusively by [ArrayHash](struct.ArrayHash.html), it should
    /// accept `key` and `value` instead of single params. 
    /// 
    /// # Parameters
    /// `key` - A key of value to be added. 
    /// `value` - A value 
    pub fn push_back(&mut self, key: K, value: V) {
        let size = core::mem::size_of::<K>();
        for b in size.to_le_bytes().iter() {
            // Put prefix of new entry as little endian bytes
            self.data.enqueue(*b);
        }
        unsafe {
            let key_ptr: *const u8 = (&key as *const K).cast();
            for i in 0..size {
                // Copy byte representation of K into this Vec
                self.data.enqueue(*(key_ptr.add(i)));
            }
            // According to hat_trie paper, the value shall be concatenated to key
            let value_ptr: *const u8 = (&value as *const V).cast();
            for i in 0..self.value_bound {
                // Copy byte representation of K into this Vec
                self.data.enqueue(*(value_ptr.add(i)));
            }
        }
        // Need to maintain key/value for as long as this Array live.
        // This is because sometime, K/V is a smart pointer type.
        // Dropping key will cause a retrieve back point to invalid memory space.
        core::mem::forget(key);
        core::mem::forget(value);
    }

    /// Push a compound key and a value into back of this slot.
    /// This function is useful for a case where key is of type `Iterator` and `ExactSizeIterator`.
    /// 
    /// It use unsafe Rust to transmute each key return by iterator and concatenate it together as
    /// sequence of bytes. It then transmute value into byte and concatenated to the key.
    /// 
    /// # Parameters
    /// `key` - An object that implement `Iterator` and `ExactSizeIterator`
    /// `value` - A value associated with the key
    pub fn push_compound_back<I>(&mut self, key: I, value: V) where I: ExactSizeIterator + Iterator<Item=K> {
        let size = core::mem::size_of::<K>() * key.len();

        // Size of compound key is equals to number of element in key * size of each key
        for b in size.to_le_bytes().iter() {
            // Put prefix of new entry as little endian bytes
            self.data.enqueue(*b);
        }
        unsafe {
            for k in key {
                let key_ptr: *const u8 = (&k as *const K).cast();
                for i in 0..core::mem::size_of::<K>() {
                    // Copy byte representation of K into this Vec
                    self.data.enqueue(*(key_ptr.add(i)));
                }
                // Need to maintain key for as long as this Array live.
                // This is because sometime, K is a smart pointer type.
                // Dropping it will cause a retrieve back point to invalid memory space.
                core::mem::forget(k);
            }
            // According to hat_trie paper, the value shall be concatenated to key
            let value_ptr = core::mem::transmute::<&V, &u8>(&value) as *const u8;
            for i in 0..self.value_bound {
                // Copy byte representation of K into this Vec
                self.data.enqueue(*(value_ptr.add(i)));
            }
        }
        // Need to maintain value for as long as this Array live.
        // This is because sometime, V is a smart pointer type.
        // Dropping it will cause a retrieve back point to invalid memory space.
        core::mem::forget(value);
    }

    /// Remove value from front of this slot.
    /// It return key/value wrap inside `Box<K>` and `Box<V>` respectively.
    /// This is because in Rust, you cannot return deref-pointer without copy.
    /// To prevent copy, we need to wrap it inside `Box`.
    /// 
    /// This operation may cause the buffer become non-contiguous.
    /// 
    /// # Return
    /// `(KeyIterator<K>, V)` - A tuple of keys iterator and value.
    pub fn pop_front(&mut self) -> Option<(KeyIterator<K>, V)> {
        if self.data.len() == 0 {
            return None
        }
        use std::convert::TryInto;
        let n = self.data.dequeue_to(core::mem::size_of::<usize>()).unwrap();

        let data_size = usize::from_le_bytes(n.as_slice().try_into().unwrap());
        let data_unit_size = core::mem::size_of::<K>();
        unsafe {
            let raw_key = self.data.dequeue_to(data_size).unwrap();
            let raw_value = self.data.dequeue_to(self.value_bound).unwrap();
            
            let key = KeyIterator {
                _k: core::marker::PhantomData,
                cursor: 0,
                key: raw_key.into(),
                len: data_size / data_unit_size
            };
            let value = core::ptr::read(raw_value.as_ptr().cast());

            Some((key, value))
        }
    }

    /// Remove given key along with its' value from this slot and return tuple of Key and value to caller.
    /// 
    /// # Parameter
    /// `k` - A reference to `K` to be remove out of this slot
    /// 
    /// # Return
    /// `Option<(K, V)>` where `Some((K, V))` is an element in this slot that has
    /// key matched with given `k`. Otherwise, it return `None`
    pub fn remove(&mut self, k: &K) -> Option<(K, V)> where K: PartialEq {
        let key_len = core::mem::size_of::<K>();
        unsafe {
            let ptr = self.data.as_ptr();
            let head = self.data.head();
            let mut i = 0;

            while i < self.data.len() {
                let key_size: usize = *ptr.add((i + head) % self.data.capacity()).cast();
                let record_ptr = ptr.add((i + head + core::mem::size_of::<usize>()) % self.data.capacity());

                // Key comparison may be expensive, we check for key_len to skip unnecessary comparison first
                if key_size == key_len && *record_ptr.cast::<K>() == *k {
                    // Key match
                    
                    // Need to use ptr::read to allow proper drop on `bytes` vec
                    let key = core::ptr::read(record_ptr.cast());
                    let value = core::ptr::read(record_ptr.add(key_size).cast());
                    self.data.silent_remove_within(i..(i + core::mem::size_of::<usize>() + key_size + core::mem::size_of::<V>())).unwrap();
                    return Some((key, value))
                } else {
                    // Key mismatch
                    i += core::mem::size_of::<usize>() + key_size + core::mem::size_of::<V>();
                }
            }

            None
        }
    }

    /// Get a pointer that point to the value that has exactly keys matched to given parameter.
    /// This is useful in case where you want to have an access directly to memory of value.
    /// 
    /// This method is unsafe because the pointer it return to is not validate. It may invalid pointer
    /// which cause seg-fault when deref because it may point to non-contiguous memory area where
    /// part of it is on a tail of buffer but subsequence part of it is on the front of buffer.
    /// 
    /// To ensure that the pointer is valid, caller must ensure that the underlying buffer is contiguous.
    /// 
    /// # Parameter
    /// `keys` - A slice of keys to look up for
    /// 
    /// # Return
    /// `Some(*const V)` if the matched keys is found. Otherwise, None.
    unsafe fn value_ptr(&self, keys: &[K]) -> Option<*const V> where K: PartialEq {
        let cursor = self.data.head();
        let tail = self.data.tail();

        while cursor < tail {
            let keys_len : usize = *self.data.as_ptr().cast();
            let cur_keys: &[K] = core::slice::from_raw_parts(self.data.as_ptr().add((cursor + core::mem::size_of::<usize>()) % self.data.capacity()).cast(), keys_len);
            if cur_keys == keys {
                return Some(self.data.as_ptr().add((cursor + core::mem::size_of::<usize>() + keys_len) % self.data.capacity()).cast())
            }
        }
        None
    }

    /// Get a pointer that point to the value that has exactly keys matched to given parameter.
    /// This is useful in case where you want to have an access directly to memory of value.
    /// 
    /// # Parameter
    /// `keys` - A slice of keys to look up for
    /// 
    /// # Return
    /// `Some(*const V)` if the matched keys is found. Otherwise, None.
    pub fn value_of<'a>(&'a self, keys: &[K]) -> Option<&'a V> where K: PartialEq {
        let cursor = self.data.head();
        let tail = self.data.tail();

        unsafe {
            while cursor < tail {
                let keys_len : usize = *self.data.as_ptr().cast();
                let cur_keys: &[K] = core::slice::from_raw_parts(self.data.as_ptr().add((cursor + core::mem::size_of::<usize>()) % self.data.capacity()).cast(), keys_len);
                if cur_keys == keys {
                    return Some(
                        core::mem::transmute::<*const V, &'a V>(self.data.as_ptr().add((cursor + core::mem::size_of::<usize>() + keys_len) % self.data.capacity()).cast())
                    )
                }
            }
        }
        None
    }

    /// Remove given compound key along with its' value from this slot 
    /// and return tuple of keys vec and value to caller.
    /// 
    /// # Parameter
    /// `keys` - A slice of `K`, which is compound key, to be remove out of this slot
    /// 
    /// # Return
    /// `Option<(Vec<K>, V)>` where `Some((Vec<K>, V))` is an element in this slot that has
    /// key matched with given `keys`. Otherwise, it return `None`
    pub fn remove_compound(&mut self, keys: &[K]) -> Option<(Vec<K>, V)> where K: PartialEq {
        let key_len = keys.len();
        unsafe {
            let ptr = self.data.as_ptr();
            let head = self.data.head();
            let mut i = 0;

            while i < self.data.len() {
                let key_size_bytes = *ptr.add((i + head) % self.data.capacity()).cast::<usize>();
                let key_size = key_size_bytes / core::mem::size_of::<K>();
                let record_ptr = ptr.add((i + head + core::mem::size_of::<usize>()) % self.data.capacity());

                // Key comparison may be expensive, we check for key_len to skip unnecessary comparison first
                if key_size == key_len && core::slice::from_raw_parts(record_ptr.cast::<K>(), key_size) == keys {
                    // Use unsafe way to prevent double copy from inner buffer of bytes into `Vec<u8>` then
                    // copy again into `Vec<K>`
                    let mut keys: Vec<K> = Vec::with_capacity(key_size);
                    keys.set_len(key_size);
                    // Copy bytes from pointer into `Vec<K>` to take ownership of those bytes into it own vec
                    core::ptr::copy(record_ptr.cast(), keys.as_mut_ptr(), key_size);
                    // Use `ptr::read` to take ownership of that bytes
                    let value = core::ptr::read(record_ptr.add(key_size).cast());

                    // Now it is safe to remove those bytes from memory as it is now copied and own by each
                    // individual which will return to caller.
                    self.data.silent_remove_within(i..(i + core::mem::size_of::<usize>() + key_size_bytes + core::mem::size_of::<V>())).unwrap();
                    return Some((keys, value))
                } else {
                    // Key mismatch
                    i += core::mem::size_of::<usize>() + key_size_bytes + self.value_bound;
                }
            }

            None
        }
    }

    /// Replace existing value of given key with new value and return old value in `Ok(V)` if
    /// given key exists, otherwise, return `Err(V)`.
    pub fn replace(&mut self, key: &K, value: V) -> Result<V, V> where K: PartialEq {
        let key_len = 1;
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let head = self.data.head();
            let mut i = 0;

            while i < self.data.len() {
                let key_size_bytes = *ptr.add((i + head) % self.data.capacity()).cast::<usize>();
                let key_size = key_size_bytes / core::mem::size_of::<K>();
                let record_ptr = ptr.add((i + head + core::mem::size_of::<usize>()) % self.data.capacity());

                // Key comparison may be expensive, we check for key_len to skip unnecessary comparison first
                if key_size == key_len && *record_ptr.cast::<K>() == *key {
                    let value_ptr = record_ptr.add(key_size_bytes).cast();
                    // Use `ptr::read` to take ownership of that bytes
                    let old_value = core::ptr::read(value_ptr);
                    core::ptr::write(value_ptr, value);

                    return Ok(old_value)
                } else {
                    // Key mismatch
                    i += core::mem::size_of::<usize>() + key_size_bytes + self.value_bound;
                }
            }

            Err(value)
        }
    }

    /// Replace existing value of given keys with new value and return old value in `Ok(V)` if
    /// given key exists, otherwise, return `Err(V)`.
    pub fn replace_compound(&mut self, keys: &[K], value: V) -> Result<V, V> where K: PartialEq {
        let key_len = keys.len();
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let head = self.data.head();
            let mut i = 0;

            while i < self.data.len() {
                let key_size_bytes = *ptr.add((i + head) % self.data.capacity()).cast::<usize>();
                let key_size = key_size_bytes / core::mem::size_of::<K>();
                let record_ptr = ptr.add((i + head + core::mem::size_of::<usize>()) % self.data.capacity());

                // Key comparison may be expensive, we check for key_len to skip unnecessary comparison first
                if key_size == key_len && core::slice::from_raw_parts(record_ptr.cast::<K>(), key_size) == keys {
                    let value_ptr = record_ptr.add(key_size_bytes).cast();
                    // Use `ptr::read` to take ownership of that bytes
                    let old_value = core::ptr::read(value_ptr);
                    core::ptr::write(value_ptr, value);

                    return Ok(old_value)
                } else {
                    // Key mismatch
                    i += core::mem::size_of::<usize>() + key_size_bytes + self.value_bound;
                }
            }

            Err(value)
        }
    }

    /// Attempt to drain this slot that make given predicate return true.
    /// This method returns [ArraySlotDrainIter](struct.ArraySlotDrainIter.html) that on each
    /// iteration, return an entry that satisfy the predicate. It will attempt to remove the element
    /// after it was return on subsequence iteration or when it is dropped.
    /// 
    /// If the iterator is drop immediately without any iteration, the element in this collection will
    /// not be removed.
    /// 
    /// Before calling this method, caller may need to call 
    /// [contiguous](struct.ArraySlotDrainIter.html#method.contiguous) first.
    /// 
    /// # Parameter
    /// `pred` - A predicate callback function which if return true, will remove the element from this slot
    /// and an iterator will return it on one of iteration.
    /// 
    /// # Return
    /// Ok([ArraySlotDrainIter](struct.ArraySlotDrainIter.html)) that return element that satisfy with given predicate
    /// or [NonContiguousError](struct.NonContiguousError.html) if the underlying buffer is not contiguous
    pub fn drain_with<P>(&mut self, pred: P) -> Result<ArraySlotDrainIter<'_, P, K, V>, NonContiguousError> where for<'r> P: FnMut(&'r [K], &'r V) -> bool {
        if self.is_contiguous() {
            let cursor = self.data.head();
            Ok(ArraySlotDrainIter {
                _k: core::marker::PhantomData,
                _v: core::marker::PhantomData,
                buffer: &mut self.data,
                cursor,
                predicate: pred,
                remove_len: None,
                to_be_remove: None
            })
        } else {
            Err(NonContiguousError {})
        }
    }

    /// Check if underlying data buffer is contiguous. It doesn't imply that first pointee
    /// of buffer pointer pointed to data.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.data.is_contiguous()
    }

    /// Make underlying buffer contiguous. This is necessary operation if you
    /// need to borrow key or value from this collection. Consider following memory layout:
    /// ```txt
    /// ----------------------------------
    /// | b2 |  1  | ... |  n - 1   | b1 |
    /// ----------------------------------
    ///   0     1    ...    n - 1      n
    /// ```
    /// If the key or value is represent by above memory layout, we cannot borrow it.
    /// With contiguous buffer, it's guarantee that `b2` will always follow `b1`.
    /// 
    /// This function will create new buffer and move all non-contiguous data into
    /// new buffer to make it contiguous and first pointee will always point to first
    /// value of pointer. 
    #[inline]
    pub fn contiguous(&mut self) {
        self.data.contiguous()
    }

    /// Attempt to get an Iterator for this [ArraySlot](struct.ArraySlot.html).
    /// If underlying queue is non-contiguous, it return [NonContiguousError](struct.NonContiguousError.html)
    /// 
    /// It need underlying buffer to be contiguous. Consider following memory layout:
    /// ```txt
    /// ----------------------------------
    /// | b2 |  1  | ... |  n - 1   | b1 |
    /// ----------------------------------
    ///   0     1    ...    n - 1      n
    /// ```
    /// If the key or value is represent by b1 and b2 layout, we cannot borrow it.
    /// With contiguous buffer, it's guarantee that `b2` will always follow `b1`.
    pub fn iter(&self) -> Result<ArraySlotIter<'_, K, V>, NonContiguousError> {
        if self.data.is_contiguous() {
            Ok(ArraySlotIter {
                cursor: self.data.head() % self.data.capacity(),
                last: self.data.tail() % self.data.capacity(),
                slot: self
            })
        } else {
            Err(NonContiguousError {})
        }
    }

    /// Attempt to get an Iterator for this [ArraySlot](struct.ArraySlot.html).
    /// If underlying queue is non-contiguous, it return [NonContiguousError](struct.NonContiguousError.html)
    /// 
    /// It need underlying buffer to be contiguous. Consider following memory layout:
    /// ```txt
    /// ----------------------------------
    /// | b2 |  1  | ... |  n - 1   | b1 |
    /// ----------------------------------
    ///   0     1    ...    n - 1      n
    /// ```
    /// If the key or value is represent by b1 and b2 layout, we cannot borrow it.
    /// With contiguous buffer, it's guarantee that `b2` will always follow `b1`.
    pub fn iter_mut<'a>(&'a mut self) -> Result<ArraySlotIterMut<'a, K, V>, NonContiguousError> {
        if self.data.is_contiguous() {
            Ok(ArraySlotIterMut {
                cursor: self.data.head() % self.data.capacity(),
                last: self.data.tail() % self.data.capacity(),
                slot: self
            })
        } else {
            Err(NonContiguousError {})
        }
    }
}

/// An iterator that remove and return an entry that evaluated by given predicate 
/// function and return true
#[derive(Debug)]
pub struct ArraySlotDrainIter<'a, P, K, V> where for<'r> P: FnMut(&'r [K], &'r V) -> bool {
    _k: core::marker::PhantomData<&'a K>,
    _v: core::marker::PhantomData<&'a V>,
    cursor: usize,
    predicate: P,
    to_be_remove: Option<usize>, 
    remove_len: Option<usize>,
    buffer: &'a mut QueueAlike<u8>
}

impl<'a, P, K, V> Iterator for ArraySlotDrainIter<'a, P, K, V> where for<'r> P: FnMut(&'r [K], &'r V) -> bool {
    type Item=(Vec<K>, V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.cursor < self.buffer.len() {
            unsafe {
                let pos = self.cursor;
                let current_ptr = self.buffer.as_mut_ptr().add(pos);
                let key_pos = pos + core::mem::size_of::<usize>();
                let len_bytes: usize = *current_ptr.cast();
                let len = len_bytes / core::mem::size_of::<K>();
                let last_key = key_pos + len_bytes;
                let next_cursor = last_key + core::mem::size_of::<V>();
                let keys = core::slice::from_raw_parts(self.buffer.as_ptr().add(key_pos).cast(), len);
                let value = &*self.buffer.as_mut_ptr().add(last_key).cast();
                if (self.predicate)(keys, value) {
                    if let Some(index) = self.to_be_remove {
                        let remove_len = self.remove_len.unwrap();
                        // Overwrite content at index by content at index + remove_len up to self.cursor
                        let first_idx = index + remove_len;
                        let buf_size = self.cursor - first_idx;
                        
                        // Move data to fill in the drained element only if there's data to be moved
                        if buf_size > 0 {
                            let mut buffer = Vec::with_capacity(buf_size);
                            buffer.set_len(buf_size);
                            let org_ptr = self.buffer.as_mut_ptr().add(first_idx);
                            core::ptr::copy(org_ptr, buffer.as_mut_ptr(), buf_size);
                            core::ptr::copy(buffer.as_mut_ptr(), self.buffer.as_mut_ptr().add(index), buf_size);

                            // Update to_be_remove and remove_len
                            self.to_be_remove = Some(index + buf_size);
                            self.remove_len = Some(next_cursor - self.cursor);
                        } else {
                            // If there's no data to replace the drained element, 
                            // it mean that next to previous drained element is also
                            // drained. The remove_len need to be added.
                            self.remove_len = Some(remove_len + next_cursor - self.cursor);
                        }
                    } else {
                        // First item found
                        self.to_be_remove = Some(self.cursor);
                        self.remove_len = Some(next_cursor - self.cursor);
                    }

                    self.cursor = next_cursor;
                    // Return keys/value
                    let mut keys: Vec<K> = Vec::with_capacity(len);
                    keys.set_len(len);
                    // Take ownership of these bytes because these bytes will be overwritten later without properly calling drop on it
                    core::ptr::copy(self.buffer.as_ptr().add(key_pos).cast(), keys.as_mut_ptr(), len);
                    let value = core::ptr::read(self.buffer.as_ptr().add(last_key).cast());
                    return Some((keys, value))
                } else {
                    self.cursor = next_cursor;
                }
            }
        }
        None
    }
}

/// Remove all elements which were return by this iterator. It do so by allocate a new heap. Copy all
/// elements which are not in list of removed element
impl<'a, P, K, V> Drop for ArraySlotDrainIter<'a, P, K, V> where for<'r> P: FnMut(&'r [K], &'r V) -> bool {
    fn drop(&mut self) {
        if let Some(index) = self.to_be_remove {
            // Properly overwritten drained element to prevent double owner of the same bytes

            let remove_len = self.remove_len.unwrap();
            // Overwrite content at index by content at index + remove_len up to self.cursor
            let first_idx = index + remove_len;
            let buf_size = self.buffer.tail() - first_idx;
            
            unsafe {
                if buf_size > 0 {
                    // only copy if needed
                    let mut buffer = Vec::with_capacity(buf_size);
                    buffer.set_len(buf_size);
                    let org_ptr = self.buffer.as_mut_ptr().add(first_idx);
                    core::ptr::copy(org_ptr, buffer.as_mut_ptr(), buf_size);
                    core::ptr::copy(buffer.as_mut_ptr(), self.buffer.as_mut_ptr().add(index), buf_size);
                }
                self.buffer.silent_remove_within((index + buf_size)..).unwrap();
            }
        } 
    }
}

impl<K, V> Drop for ArraySlot<K, V> {
    fn drop(&mut self) {
        while self.data.len() > 0 {
            self.pop_front();
        }
    }
}

pub struct KeyIterator<K> {
    _k: core::marker::PhantomData<K>,
    cursor: usize,
    key: QueueAlike<u8>,
    len: usize
}

impl<K> KeyIterator<K> {
    /// Get a [RefKeyIterator](struct.RefKeyIterator.html) with cursor at current iterator position
    pub fn as_ref(&self) -> RefKeyIterator<'_, K> {
        RefKeyIterator {
            _k: core::marker::PhantomData,
            cursor: self.cursor,
            keys: &self.key,
            last: self.key.tail(),
            len: self.len
        }
    }
    /// Get a [RefKeyIterMut](struct.RefKeyIterMut.html) with cursor at current iterator position
    pub fn as_mut(&mut self) -> RefKeyIterMut<'_, K> {
        RefKeyIterMut {
            _k: core::marker::PhantomData,
            cursor: self.cursor,
            keys: self.key.as_mut_ptr(),
            last: self.key.tail(),
            len: self.len
        }
    }
    /// Convert what remain in this iterator into `Vec<K>`
    pub fn to_vec(mut self) -> Vec<K> {
        let remain = self.len - self.cursor;
        let n = remain / core::mem::size_of::<K>();
        let mut result = Vec::with_capacity(n);
        unsafe {
            result.set_len(n);
            core::ptr::copy(self.key.as_ptr().cast(), result.as_mut_ptr(), n);
            self.key.silent_remove_within(..self.len).unwrap();
        }

        result
    }
}

impl<K> Iterator for KeyIterator<K> {
    type Item=K;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.len {
            unsafe {
                let key = self.key.as_ptr().add(self.key.head());
                let result = core::ptr::read(key.cast());
                self.key.silent_remove_within(..core::mem::size_of::<K>()).unwrap();
                self.cursor += 1;

                Some(result)
            }
        } else {
            None
        }
    }
}

impl<K> core::iter::FusedIterator for KeyIterator<K> {}
impl<K> core::iter::ExactSizeIterator for KeyIterator<K> {
    fn len(&self) -> usize {
        self.len
    }
}

/// KeyIterator maybe drop without iterating so we need to ensure that every key
/// is dropped to prevent memory leak. Since each iteration dequeues from internal buffer,
/// it is safe to assume that every remaining object in this buffer can be drop
impl<K> Drop for KeyIterator<K> {
    fn drop(&mut self) {
        if self.key.len() > 0 {
            unsafe {
                core::ptr::drop_in_place(
                    core::ptr::slice_from_raw_parts_mut(self.key.as_mut_ptr() as *mut K, self.len)
                );
            }
        }
    }
}

/// An iterator over each entry in this slot. It will return tuple of [RefKeyIterator](struct.RefKeyIterator.html)
/// and value `V`
#[derive(Debug)]
pub struct ArraySlotIntoIter<K, V> {
    cursor: usize,
    last: usize,
    slot: ArraySlot<K, V>
}

impl<K, V> Iterator for ArraySlotIntoIter<K, V> {
    type Item=(KeyIterator<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.last {
            return None
        }

        self.slot.pop_front()
        // unsafe {
        //     let size: usize = *self.slot.data.as_ptr().add(self.cursor).cast();
        //     let key_offset = self.cursor + core::mem::size_of::<usize>();
        //     let mut key = Vec::with_capacity(size);
        //     key.set_len(size);
        //     let buffer = key.as_mut_ptr();
        //     core::ptr::copy(self.slot.data.as_ptr().add(key_offset).cast(), buffer, size / core::mem::size_of::<K>());
        //     // let key = KeyIterator {
        //     //     _k: core::marker::PhantomData,
        //     //     cursor: key_offset,
        //     //     len: size,
        //     //     key: self.slot.data
        //     // };
        //     let value_offset = key_offset + size;
        //     let value = core::ptr::read(self.slot.data.as_ptr().add(value_offset).cast());
        //     self.cursor = value_offset + core::mem::size_of::<V>();
        //     return Some((key, value))
        // }
    }
}

impl<K, V> core::iter::FusedIterator for ArraySlotIntoIter<K, V> {}

/// Consume self and return an iterator over each element in slot.
/// This operation may or may not slow depending on whether the slot is contiguous.
/// If it's contiguous, it'll immediately return an iterator. Otherwise,
/// it will attempt to make the slot contiguous first.
impl<K, V> IntoIterator for ArraySlot<K, V> {
    type Item = (KeyIterator<K>, V);
    type IntoIter = ArraySlotIntoIter<K, V>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.contiguous();
        ArraySlotIntoIter {
            cursor: 0,
            last: self.data.tail(),
            slot: self
        }
    }
}

/// An iterator that return borrow ref by using unsafe transmute on raw byte
/// memory stored in buffer.
pub struct RefKeyIterator<'a, K> {
    _k: core::marker::PhantomData<&'a K>,
    cursor: usize,
    keys: &'a QueueAlike<u8>,
    last: usize,
    len: usize,
}

impl<'a, K> Iterator for RefKeyIterator<'a, K> {
    type Item=&'a K;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.last {

            unsafe {
                let key = self.keys.as_ptr().add(self.cursor).cast();
                let result = core::mem::transmute::<*const K, &'a K>(key);
                self.cursor += core::mem::size_of::<K>();

                Some(result)
            }
        } else {
            None
        }
    }
}

impl<'a, K> core::iter::FusedIterator for RefKeyIterator<'a, K> {
}

impl<'a, K> ExactSizeIterator for RefKeyIterator<'a, K> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K> RefKeyIterator<'a, K> {
    /// Turn this iterator into a slice of keys
    pub fn as_slice(&self) -> &[K] {
        unsafe {
            debug_assert!(self.cursor < self.last);
            core::slice::from_raw_parts(self.keys.as_ptr().add(self.cursor).cast(), self.len)
        }
    }
}

/// An iterator over each entry in this slot. It will return tuple of [RefKeyIterator](struct.RefKeyIterator.html)
/// and value `V`
#[derive(Debug)]
pub struct ArraySlotIter<'a, K, V> {
    cursor: usize,
    last: usize,
    slot: &'a ArraySlot<K, V>
}

impl<'a, K, V>  Iterator for ArraySlotIter<'a, K, V> {
    type Item=(RefKeyIterator<'a, K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        use core::convert::TryInto;

        if self.cursor >= self.last {
            return None
        }

        unsafe {
            let size = usize::from_le_bytes(core::slice::from_raw_parts(self.slot.data.as_ptr().add(self.cursor), core::mem::size_of::<usize>()).try_into().unwrap());
            let key_offset = self.cursor + core::mem::size_of::<usize>();
            let key = RefKeyIterator {
                _k: core::marker::PhantomData,
                cursor: key_offset,
                last: key_offset + size,
                len: size / core::mem::size_of::<K>(),
                keys: &self.slot.data
            };
            let value_offset = key_offset + size;
            let value = core::mem::transmute::<*const V, &'a V>(self.slot.data.as_ptr().add(value_offset).cast());
            self.cursor = value_offset + core::mem::size_of::<V>();
            return Some((key, value))
        }
    }
}

impl<'a, K, V> core::iter::FusedIterator for ArraySlotIter<'a, K, V> {}

/// An iterator that return mutably borrow ref by using unsafe transmute on raw byte
/// memory stored in buffer.
pub struct RefKeyIterMut<'a, K> {
    _k: core::marker::PhantomData<&'a mut K>,
    cursor: usize,
    keys: *mut u8,
    last: usize,
    len: usize,
}

impl<'a, K> Iterator for RefKeyIterMut<'a, K> {
    type Item=&'a mut K;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.last {

            unsafe {
                let key = self.keys.add(self.cursor).cast();
                let result = core::mem::transmute::<*mut K, &'a mut K>(key);
                self.cursor += core::mem::size_of::<K>();

                Some(result)
            }
        } else {
            None
        }
    }
}

impl<'a, K> core::iter::FusedIterator for RefKeyIterMut<'a, K> {
}

impl<'a, K> ExactSizeIterator for RefKeyIterMut<'a, K> {
    fn len(&self) -> usize {
        self.len
    }
}

/// An iterator over each entry in this slot. It will return tuple of [RefKeyIterMut](struct.RefKeyIterMut.html)
/// and value `V`
#[derive(Debug)]
pub struct ArraySlotIterMut<'a, K, V> 
{
    cursor: usize,
    last: usize,
    slot: &'a mut ArraySlot<K, V>
}

impl<'a, K, V>  Iterator for ArraySlotIterMut<'a, K, V> 
{
    type Item=(RefKeyIterMut<'a, K>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        use core::convert::TryInto;
        if self.cursor >= self.last {
            return None
        }

        unsafe {
            let size = usize::from_le_bytes(core::slice::from_raw_parts(self.slot.data.as_ptr().add(self.cursor), core::mem::size_of::<usize>()).try_into().unwrap());
            let key_offset = self.cursor + core::mem::size_of::<usize>();
            let key = RefKeyIterMut {
                _k: core::marker::PhantomData,
                cursor: key_offset,
                last: key_offset + size,
                len: size / core::mem::size_of::<K>(),
                keys: self.slot.data.as_mut_ptr()
            };
            let value_offset = key_offset + size;
            let value = core::mem::transmute::<*mut V, &'a mut V>(self.slot.data.as_mut_ptr().add(value_offset).cast());
            self.cursor = value_offset + core::mem::size_of::<V>();
            return Some((key, value))
        }
    }
}

impl<'a, K, V> core::iter::FusedIterator for ArraySlotIterMut<'a, K, V> {}

pub struct NonContiguousError {}

impl core::fmt::Debug for NonContiguousError {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("The operation cannot be perform with non-contiguous data collection")
    }
}


/// An implementation of ArrayHash in pure Rust.
/// 
/// ArrayHash is a data structure where the index is determined by hash and
/// each entry in array is a `Vec` that store data and all it collision.
/// 
/// Oritinal paper can be found [here](Askitis, N. & Zobel, J. (2005), Cache-conscious collision resolution for string hash tables, in ‘Proc. SPIRE String Processing and Information Retrieval Symp.’, Springer-Verlag, pp. 92–104)
/// 
/// In this implementation, user can supplied their own choice of hasher but it need to implement `Clone`.
/// 
/// The data can be anything that implement `Hash` and `Clone`. This is due to nature of
/// `Vec` that the first allocation need to be cloneable. Otherwise, it cann't be pre-allocate.
/// 
/// The default `Hasher`, if not provided, will be `XxHash64`.
#[derive(Clone, Debug)]
pub struct ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + PartialEq + Clone, V: Clone {
    buckets: Option<Box<[ArraySlot<K, V>]>>,
    hasher: H,
    capacity: usize,
    max_load_factor: usize,
    size: usize
}

impl<H, K, V> ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + PartialEq + Clone, V: Clone {
    /// Add or replace entry into this `HashMap`.
    /// If entry is replaced, it will be return in `Option`.
    /// Otherwise, it return `None`
    /// 
    /// # Parameter
    /// - `key` - A key of data
    /// - `value` - A value of data
    /// 
    /// # Return
    /// Option that contains tuple of (key, value) that got replaced or `None` if it is
    /// new entry
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        let mut index = self.make_key(&key);
        match self.buckets.as_mut().unwrap()[index].replace(&key, value) {
            Ok(old_value) => Some(old_value),
            Err(value) => {
                self.size += 1;

                if self.maybe_expand() {
                    index = self.make_key(&key);
                }
    
                self.buckets.as_mut().unwrap()[index].push_back(key, value);
                None
            }
        }
    }
    
    /// Add or replace entry into this `HashMap`.
    /// If entry is replaced, it will be return in `Option`.
    /// Otherwise, it return `None`
    /// 
    /// # Parameter
    /// - `keys` - A vec of keys of data
    /// - `value` - A value of data
    /// 
    /// # Return
    /// Option that contains tuple of (key, value) that got replaced or `None` if it is
    /// new entry
    pub fn put_compound(&mut self, keys: Vec<K>, value: V) -> Option<V> {
        let mut index = self.make_compound_key(keys.iter());
        
        match self.buckets.as_mut().unwrap()[index].replace_compound(keys.as_slice(), value) {
            Ok(old_value) => Some(old_value),
            Err(value) => {
                self.size += 1;

                if self.maybe_expand() {
                    index = self.make_compound_key(keys.iter());
                }
    
                self.buckets.as_mut().unwrap()[index].push_compound_back(keys.into_iter(), value);
                None
            }
        }
    }

    /// Try to put value into this `ArrayHash`.
    /// If the given key is already in used, leave old entry as is and return current value associated with the key.
    /// Otherwise, add entry to this `ArrayHash` and return None.
    /// 
    /// # Parameter
    /// - `key` - A key to be add to this collection.
    /// - `value` - A value to be add to this collection
    /// # Return
    /// If key already exist, it return `Some(&V, K, V)` where `&V` is current value
    /// associated with the given key, `K` and `V are paremeters supplied to this method. 
    /// If key is not exist, it put the value into this `ArrayHash` and return None.
    pub fn try_put<'a>(&'a mut self, key: K, value: V) -> Option<(&'a V, K, V)> where K: PartialEq {
        let mut index = self.make_key(&key);

        unsafe {
            let keys = core::slice::from_raw_parts(&key as *const K, 1);
            // It is safe to use value_ptr here because this ArrayHash use pop_front only on drain 
            // which is guarantee to remove all the data
            match self.buckets.as_ref().unwrap()[index].value_ptr(keys) {
                Some(v_ptr) => Some((core::mem::transmute::<*const V, &'a V>(v_ptr), key, value)),
                None => {
                    self.size += 1;
                    if self.maybe_expand() {
                        index = self.make_key(&key);
                    }
                    self.buckets.as_mut().unwrap()[index].push_back(key, value);
                    None
                }
            }
        }
    }

    /// Try to put value into this `ArrayHash`.
    /// If the given keys is already in used, leave old entry as is and return current 
    /// reference to value associated with the key along with given `keys` and `value`
    /// parameter.
    /// Otherwise, add entry to this `ArrayHash` and return None.
    /// 
    /// # Parameter
    /// - `keys` - A vec of keys to be add to this collection.
    /// - `value` - A value to be add to this collection
    /// # Return
    /// If key already exist, it return `Some(&V, Vec<K>, V)` where `&V` is current value
    /// associated with the given keys, `Vec<K>` and `V are paremeters supplied to this method. 
    /// If key is not exist, it put the value into this `ArrayHash` and return None.
    pub fn try_put_compound<'a>(&'a mut self, keys: Vec<K>, value: V) -> Option<(&'a V, Vec<K>, V)> where K: PartialEq {
        let mut index = self.make_compound_key(keys.iter());

        unsafe {
            // It is safe to use value_ptr here because this ArrayHash use pop_front only on drain 
            // which is guarantee to remove all the data
            match self.buckets.as_ref().unwrap()[index].value_ptr(keys.as_slice()) {
                Some(v_ptr) => Some((core::mem::transmute::<*const V, &'a V>(v_ptr), keys, value)),
                None => {
                    self.size += 1;
                    if self.maybe_expand() {
                        index = self.make_compound_key(keys.iter());
                    }
                    self.buckets.as_mut().unwrap()[index].push_compound_back(keys.into_iter(), value);
                    None
                }
            }
        }
    }

    /// Get a value of given key from this `ArrayHash`
    /// 
    /// # Parameter
    /// - `key` - A key to look for. It can be anything that can be deref into K
    /// 
    /// # Return
    /// An `Option` contains a value or `None` if it is not found.
    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self.make_key(key);
        let slot = &self.buckets.as_ref().unwrap()[index];

        for (ref mut k, ref v) in slot.iter().unwrap() {
            if k.len() == 1 && *k.next().unwrap() == *key {
                return Some(v)
            }
        }
        None
    }

    /// Get a value using deref type.
    /// 
    /// This is usable only if the key is a type of smart pointer that can be deref into another type
    /// which implement `Hash` and `PartialEq`.
    /// 
    /// For example, if K is `Box<[u8]>`, you can use `&[u8]` to query for a value
    /// 
    /// # Parameter
    /// `key` - Any type that implement `Deref` where type after `Deref` is that same type
    /// as actual type of `key` beneath type `K`.
    /// 
    /// # Return
    /// `Some(&V)` if key exist in this table. Otherwise None.
    pub fn smart_get<T, Q>(&self, key: Q) -> Option<&V> where Q: core::ops::Deref<Target=T>, K: core::ops::Deref<Target=T>, T: core::hash::Hash + core::cmp::PartialEq + ?Sized {
        let mut local_hasher = self.hasher.clone();
        key.hash(&mut local_hasher);
        let index = local_hasher.finish() as usize % self.capacity;

        let slot = &self.buckets.as_ref().unwrap()[index];

        for (ref mut k, ref v) in slot.iter().unwrap() {
            if k.len() == 1 && **k.next().unwrap() == *key {
                return Some(v)
            }
        }
        None
    }

    /// Get a value of given compound key from this `ArrayHash`
    /// 
    /// # Parameter
    /// - `keys` - A slice of keys to look for. 
    /// 
    /// # Return
    /// An `Option` contains a value or `None` if it is not found.
    pub fn get_compound(&self, keys: &[K]) -> Option<&V> {
        let index = self.make_compound_key(keys.iter());
        let slot = &self.buckets.as_ref().unwrap()[index];

        for (ref k, ref v) in slot.iter().unwrap() {
            if k.as_slice() == keys {
                return Some(v)
            }
        }
        None
    }

    /// Attempt to remove entry with given key from this `ArrayHash`.
    /// 
    /// # Parameter
    /// - `key` - A key of entry to be remove. 
    /// 
    /// # Return
    /// Option that contain tuple of (key, value) or `None` of key is not found
    pub fn remove(&mut self, key: &K) -> Option<(K, V)> {
        let slot_idx = self.make_key(key);
        let slot = self.buckets.as_mut().unwrap();
        let result = slot[slot_idx].remove(key);
        if let Some(entry) = result {
            self.size -= 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Attempt to remove entry with given key from this `ArrayHash`.
    /// 
    /// # Parameter
    /// - `keys` - A slice of keys of entry to be remove.
    /// 
    /// # Return
    /// Option that contain tuple of (key, value) or `None` of key is not found
    pub fn remove_compound(&mut self, keys: &[K]) -> Option<(Vec<K>, V)> {
        let slot_idx = self.make_compound_key(keys.iter());
        let slot = self.buckets.as_mut().unwrap();
        let result = slot[slot_idx].remove_compound(keys);
        if let Some(entry) = result {
            self.size -= 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Make every buffer in this collection contiguous.
    /// Any iterator over this collection need underlying buffer to be contiguous.
    /// Otherwise, it'll panic while iterating.
    /// 
    /// [drain](struct.ArrayHash.html#method.drain) is a special case where it doesn't
    /// need underlying buffer to be continguous. This is because every data will
    /// eventually remove out and the order of removal is from start to finish.
    /// [drain_with](struct.ArrayHash.html#method.drain_with) doesn't have that luxury.
    /// It requires underlying buffer to be contiguous.
    /// 
    /// It is make as an optional because if user of this collection never
    /// call [remove](struct.ArrayHash.html#method.remove) method then the 
    /// collection will always continguous.
    /// 
    /// This operation is usually expensive. You should try to minimize calling
    /// this method.
    pub fn contiguous(&mut self) {
        for slot in self.buckets.as_mut().unwrap().iter_mut() {
            slot.contiguous();
        }
    }

    /// Current number of entry in this `ArrayHash`
    pub fn len(&self) -> usize {
        self.size
    }

    /// Get an iterator over this `ArrayHash`. 
    /// 
    /// If the collection has some element removed by [remove](struct.ArrayHash.html#method.remove)
    /// or [remove_compound](struct.ArrayHash.html#method.remove_compound), 
    /// you may need to call [contiguous](struct.ArrayHash.html#method.contiguous) method 
    /// to prevent iterator to panic while iterating.
    /// 
    /// # Return
    /// [ArrayHashIterator](struct.ArrayHashIterator.html) that return reference
    /// to each entry in this `ArrayHash`
    pub fn iter(&self) -> ArrayHashIterator<'_, K, V> {
        let slots = self.buckets.as_ref().unwrap();
        ArrayHashIterator {
            buckets: &slots,
            current_iterator: slots[0].iter().unwrap(),
            remain_slots: slots.len() - 1, // it need to - 1 as one of it is considered in process
            slot_cursor: 0,
            size: self.size
        }
    }

    /// Get a mutable iterator over this `ArrayHash`.
    /// 
    /// If the collection has some element removed by [remove](struct.ArrayHash.html#method.remove)
    /// or [remove_compound](struct.ArrayHash.html#method.remove_compound), 
    /// you may need to call [contiguous](struct.ArrayHash.html#method.contiguous) method 
    /// to prevent panic on invoking this method.
    /// 
    /// Warning, you shall not modify the key part of entry. If you do, it might end up
    /// accessible only by iterator but not with [get](struct.ArrayHash.html#method.get).
    /// 
    /// # Return
    /// [ArrayHashIterMut](struct.ArrayHashIterMut.html) that return mutably reference
    /// to each entry in this `ArrayHash`
    pub fn iter_mut(&mut self) -> ArrayHashIterMut<'_, K, V> {
        if self.size > 0 {
            let buckets: Box<[ArraySlotIterMut<K, V>]> = self.buckets.as_mut().unwrap().iter_mut().map(|slot| {
                slot.iter_mut().unwrap()
            }).collect();
            let remain_slots = buckets.len() - 1;
            ArrayHashIterMut {
                // Only get iter_mut from entry with some element
                buckets,
                remain_slots, // similar to immutable iter, 0 index is already in process
                slot_cursor: 0,
                size: self.size
            }
        } else {
            ArrayHashIterMut {
                // Create an empty iterator so it will be called only once then finish.
                // We cannot use iter::empty() as the type is incompatible.
                buckets: vec![].into_boxed_slice(),
                remain_slots: 0,
                slot_cursor: 0,
                size: self.size
            }
        }
    }

    /// Return an iterator that drain all entry out of this [ArrayHash](struct.ArrayHash.html).
    /// 
    /// After the iterator is done, this [ArrayHash](struct.ArrayHash.html) will become empty.
    /// 
    /// This method will immediately set size to 0.
    /// 
    /// # Return
    /// [DrainIter](struct.DrainIter.html) - An iterator that will drain all element
    /// out of this [ArrayHash](struct.ArrayHash.html).
    pub fn drain(&mut self) -> DrainIter<K, V> {
        let mut bucket_iter = self.buckets.as_mut().unwrap().iter_mut();
        let current_slot = bucket_iter.next();
        self.size = 0;

        DrainIter {
            bucket_iter,
            current_slot,
            size: self.size
        }
    }

    /// Return an iterator that drain some entry out of this [ArrayHash](struct.ArrayHash.html).
    /// 
    /// After the iterator is done, this [ArrayHash](struct.ArrayHash.html) size will be shrink.
    /// 
    /// This method will return an iterator where each element it drain will cause a size deduction
    /// on this [ArrayHash](struct.ArrayHash.html).
    /// 
    /// # Return
    /// [DrainWithIter](struct.DrainWithIter.html) - An iterator that will drain all element
    /// out of this [ArrayHash](struct.ArrayHash.html).
    pub fn drain_with<P>(&mut self, pred: P) -> DrainWithIter<P, K, V> where for<'r> P: Copy + Fn(&'r [K], &'r V) -> bool {
        let mut bucket_iter = self.buckets.as_mut().unwrap().iter_mut();
        let current_slot = bucket_iter.next().unwrap().drain_with(pred).ok();
        let size = self.size; // Max size of iterator

        DrainWithIter {
            bucket_iter,
            cur_size: &mut self.size,
            current_slot,
            predicate: pred,
            size
        }
    }

    /// Split this [ArrayHash](struct.ArrayHash.html) by given predicate closure. 
    /// Every element that closure evaluate to true will be remove from this [ArrayHash](struct.ArrayHash.html)
    /// and return in new instance of [ArrayHash](struct.ArrayHash.html).
    /// 
    /// This is different from using [drain_with](struct.ArrayHash.html#method.drain_with) to drain
    /// some element into another [ArrayHash](struct.ArrayHash.html) by this method will return 
    /// [ArrayHash](struct.ArrayHash.html) with exactly identical property, i.e. Hasher, buckets_size,
    /// and max_load_factor, whereas [drain_with](struct.ArrayHash.html#method.drain_with) let
    /// caller instantiate [ArrayHash](struct.ArrayHash.html) yourself.
    /// 
    /// Since the instant it returns, use the same Hasher. It is safe to assume that all elements shall
    /// reside in the same bucket number thus this method speed up the split by ignore hashing altogether and
    /// store the entry directly into the same bucket number as in this [ArrayHash](struct.ArrayHash.html)
    /// 
    /// This method automatically make underlying buffer contiguous first before attempt to split it.
    /// 
    /// # Parameter
    /// `pred` - A closure that evaluate an entry. If it return true, the entry shall be moved into a new 
    /// [ArrayHash](struct.ArrayHash.html).
    /// 
    /// # Return
    /// An [ArrayHash](struct.ArrayHash.html) that contains all entry that `pred` evaluate to true.
    pub fn split_by<F>(&mut self, pred: F) -> ArrayHash<H, K, V> where F: Copy + Fn(&[K], &V) -> bool {
        let mut other = ArrayHashBuilder::with_hasher(self.hasher.clone())
                                      .buckets_size(self.buckets.as_ref().unwrap().len())
                                      .max_load_factor(self.max_load_factor)
                                      .build();
        let buckets = self.buckets.as_mut().unwrap();
        for i in 0..buckets.len() {
            for (keys, value) in buckets[i].drain_with(pred).unwrap() {
                self.size -= 1;
                other.buckets.as_mut().unwrap()[i].push_compound_back(keys.into_iter(), value);
                other.size += 1;
            }
        }

        other
    }

    #[inline(always)]
    fn make_key<Q>(&self, key: Q) -> usize where Q: core::ops::Deref<Target=K> {
        let mut local_hasher = self.hasher.clone();
        key.hash(&mut local_hasher);
        local_hasher.finish() as usize % self.capacity
    }

    #[inline(always)]
    fn make_compound_key<'a, I>(&self, keys: I) -> usize where I: core::iter::Iterator<Item=&'a K>, K: 'a {
        let mut local_hasher = self.hasher.clone();
        for key in keys {
            key.hash(&mut local_hasher);
        }
        local_hasher.finish() as usize % self.capacity
    }

    /// Check if it over scaling threshold. If it is, expand the bucket, rehash all the key, and
    /// put everything to it new place.
    /// # Return
    /// true if it was expanded, false if it doesn't need to expand
    #[inline(always)]
    fn maybe_expand(&mut self) -> bool {
        if self.size < self.max_load_factor {
            return false
        }

        let buckets = self.buckets.take().unwrap();
        let old_buckets = buckets.into_vec();
        let new_capacity = self.capacity * 2;
        self.capacity = new_capacity;
        self.max_load_factor *= 2;
        // Assume hash is evenly distribute entry in bucket, the new slot size shall be <= old slot size.
        // This is because the bucket size is doubled.
        let mut buckets = vec!(ArraySlot::with_capacity(old_buckets[0].data.capacity()); new_capacity);
        old_buckets.into_iter().for_each(|slot| {
            for (key, value) in slot {
                let index = self.make_compound_key(key.as_ref());
                buckets[index].push_compound_back(key, value);
            }
        });
        self.buckets = Some(buckets.into_boxed_slice());

        true
    }
}

/// An iterator that return a reference to each entry in `ArrayHash`.
/// It is useful for scanning entire `ArrayHash`.
#[derive(Debug)]
pub struct ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    buckets: &'a Box<[ArraySlot<K, V>]>,
    current_iterator: ArraySlotIter<'a, K, V>,
    slot_cursor: usize,
    remain_slots: usize,
    size: usize
}

impl<'a, K, V> Iterator for ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(RefKeyIterator<'a, K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.current_iterator.next();

        while result.is_none() {
            if self.slot_cursor < self.remain_slots {
                self.slot_cursor += 1;
                self.current_iterator = self.buckets[self.slot_cursor].iter().unwrap();
                result = self.current_iterator.next();
            } else {
                break
            }
        }

        result
    }
}

impl<'a, K, V> core::iter::FusedIterator for ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {}

impl<'a, K, V> core::iter::ExactSizeIterator for ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    fn len(&self) -> usize {
        self.size
    }
}

/// An iterator that return a mutably reference to each entry in `ArrayHash`.
/// It is useful for scanning entire `ArrayHash` to manipulate it value.
/// 
/// It can cause undesired behavior if user alter the key in place as the slot position is
/// calculated by hashed value of the key. It might endup having duplicate key on different slot and
/// anytime caller use [get method](struct.ArrayHash.html#method.get), it will always return that value instead
/// of this modified key.
/// 
/// If you need to modify key, consider [remove](struct.ArrayHash.html#method.remove) old key first then
/// [put](struct.ArrayHash.html#method.put) the new key back in.
#[derive(Debug)]
pub struct ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    buckets: Box<[ArraySlotIterMut<'a, K, V>]>,
    remain_slots: usize,
    slot_cursor: usize,
    size: usize
}

impl<'a, K, V> Iterator for ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(RefKeyIterMut<'a, K>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.buckets[self.slot_cursor].next();
        
        while result.is_none() {
            if self.slot_cursor < self.remain_slots {
                self.slot_cursor += 1;
                result = self.buckets[self.slot_cursor].next();
            } else {
                break;
            }
        }

        result
    }
}

impl<'a, K, V> core::iter::FusedIterator for ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {}

impl<'a, K, V> core::iter::ExactSizeIterator for ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    fn len(&self) -> usize {
        self.size
    }
}

#[derive(Debug)]
pub struct ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    buckets: std::vec::IntoIter<ArraySlot<K, V>>,
    current_iterator: ArraySlotIntoIter<K, V>,
    size: usize
}

impl<K, V> Iterator for ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(KeyIterator<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.current_iterator.next();

        while result.is_none() {
            if let Some(slot) = self.buckets.next() {
                self.current_iterator = slot.into_iter();
                result = self.current_iterator.next();
            } else {
                // entire ArrayHash is exhausted
                break
            }
        }

        result
    }
}

impl<K, V> core::iter::FusedIterator for ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {}

impl<K, V> core::iter::ExactSizeIterator for ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    fn len(&self) -> usize {
        self.size
    }
}

impl<H, K, V> IntoIterator for ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(KeyIterator<K>, V);
    type IntoIter=ArrayHashIntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        if self.size >= 1 {
            let mut buckets = self.buckets.unwrap().to_vec().into_iter();
            let current_iterator = buckets.next().unwrap().into_iter();
            ArrayHashIntoIter {
                buckets,
                current_iterator,
                size: self.size
            }
        } else {
            let mut emptied_bucket = vec![ArraySlot::new()].into_iter();
            let emptied_iterator = emptied_bucket.next().unwrap().into_iter();
            ArrayHashIntoIter {
                buckets: emptied_bucket,
                current_iterator: emptied_iterator,
                size: 0
            }
        }
    }
}

impl<'a, H, K, V> IntoIterator for &'a ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(RefKeyIterator<'a, K>, &'a V);
    type IntoIter=ArrayHashIterator<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, H, K, V> IntoIterator for &'a mut ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(RefKeyIterMut<'a, K>, &'a mut V);
    type IntoIter=ArrayHashIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// An iterator that will drain it underlying [ArrayHash](struct.ArrayHash.html).
#[derive(Debug)]
pub struct DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    bucket_iter: core::slice::IterMut<'a, ArraySlot<K, V>>,
    current_slot: Option<&'a mut ArraySlot<K, V>>,
    size: usize,
}

impl<'a, K, V> Iterator for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(KeyIterator<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.current_slot.as_mut().unwrap().pop_front();

        while result.is_none() {
            // Make empty array slot pointer become contiguous.
            // This is very cheap because there's no data to copy.
            self.current_slot.as_mut().unwrap().contiguous(); 

            self.current_slot = self.bucket_iter.next();
            if self.current_slot.is_some() {
                result = self.current_slot.as_mut().unwrap().pop_front();
            } else {
                break;
            }
        }

        result
    }
}

impl<'a, K, V> core::iter::FusedIterator for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {}
impl<'a, K, V> core::iter::ExactSizeIterator for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    fn len(&self) -> usize {
        self.size
    }
}

/// Make dropping DrainIter simply remove every elements from every slots
impl<'a, K, V> Drop for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    fn drop(&mut self) {
        if let Some(ref mut slot) = self.current_slot {
            slot.clear();
        }

        while let Some(ref mut slot) = self.bucket_iter.next() {
            slot.clear()
        }
    }
}
/// An iterator that remove and return element that satisfy predicate.
/// It will also update the size of borrowed [ArrayHash](struct.ArrayHash.html) on each
/// iteration.
#[derive(Debug)]
pub struct DrainWithIter<'a, P, K, V> where P: Copy + for<'r> Fn(&'r [K], &'r V) -> bool, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    bucket_iter: core::slice::IterMut<'a, ArraySlot<K, V>>,
    cur_size: &'a mut usize,
    current_slot: Option<ArraySlotDrainIter<'a, P, K, V>>,
    predicate: P,
    size: usize
}

impl<'a, P, K, V> Iterator for DrainWithIter<'a, P, K, V> where P: Copy + for<'r> Fn(&'r [K], &'r V) -> bool, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    type Item=(Vec<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ref mut v) = self.current_slot {
            if let Some(e) = v.next() {
                *self.cur_size -= 1;
                return Some(e)
            }

            loop {
                let next_slot = self.bucket_iter.next();
                if let Some(slot) = next_slot {
                    self.current_slot = slot.drain_with(self.predicate).ok();
                    break;
                } else {
                    // All slot in every buckets are evaulated now
                    return None
                }
            }
        }

        None
    }
}

/// Dropping [DrainWithIter](struct.DrainWithIter.html) will remove all elements that match
/// with predicate. Every key(s) and value will be dropped.
impl<'a, P, K, V> Drop for DrainWithIter<'a, P, K, V> where P: Copy + for<'r> Fn(&'r [K], &'r V) -> bool, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {
    fn drop(&mut self) {
        let mut count = 0;
        // Drain all element that match predicate
        if let Some(ref mut slot) = self.current_slot {
            for _ in slot {
                count += 1;
            }
        }

        while let Some(ref mut slot) = self.bucket_iter.next() {
            slot.drain_with(self.predicate).unwrap().for_each(|_| {
                count += 1;
            });
        }

        *self.cur_size -= count;
    }
}

impl<'a, P, K, V> core::iter::FusedIterator for DrainWithIter<'a, P, K, V> where P: Copy + for<'r> Fn(&'r [K], &'r V) -> bool, K: core::hash::Hash + core::cmp::PartialEq + Clone, V: Clone {}

#[cfg(test)]
mod tests;