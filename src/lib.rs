//! An implementation of ArrayHash
//! 
//! ArrayHash is a data structure where the index is determined by hash and
//! each entry in array is a `Vec` that store data and all it collision.
//! 
//! Oritinal paper can be found [here](Askitis, N. & Zobel, J. (2005), Cache-conscious collision resolution for string hash tables, in ‘Proc. SPIRE String Processing and Information Retrieval Symp.’, Springer-Verlag, pp. 92–104)
//! 
//! This implementation try to use generic wherever possible.
//! It end up with ArrayHash that take anything that is clonable as value and anything that
//! implement `Hash` and `PartialEq` as key.
//! It let you choose whichever `Hasher` that you want. The only constraint is that
//! `Hasher` must implement `Clone` trait.
//! 
//! It supports read only iteration, mutably iteration, and owned iteration.
//! 
//! To create [ArrayHash](struct.ArrayHash.html) use [ArrayHashBuilder](struct.ArrayHashBuilder.html).
//! The default `Hasher` is `XxHasher64`.

use twox_hash::{RandomXxHashBuilder64, XxHash64};
use core::hash::{BuildHasher, Hash, Hasher};
use core::borrow::Borrow;

const MAX_LOAD_FACTOR: usize = 100_000; // Number of element before resize the table

// Make each bucket fit into single memory page
const DEFAULT_BUCKETS_SIZE: usize = 4096 / std::mem::size_of::<usize>(); 
const DEFAULT_SLOT_SIZE: usize = 8;

/// A builder that use for build an [ArrayHash](struct.ArrayHash.html).
#[derive(Clone, Hash, PartialEq)]
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

/// Make [ArrayHashBuilder](struct.ArrayHashBuilder.html) with spec from existing [ArrayHash](struct.ArrayHash.html).
/// 
/// This is useful for creating an object of array hash that may be later compare to baseline array for equality.
impl<'a, H, K, V> From<&'a ArrayHash<H, K, V>> for ArrayHashBuilder<H> where H: Clone + Hasher, K: Hash + PartialEq {
    fn from(ah: &'a ArrayHash<H, K, V>) -> ArrayHashBuilder<H> {
        let buckets = ah.buckets.as_ref().unwrap();
        
        debug_assert!(buckets.len() > 0);

        ArrayHashBuilder {
            buckets_size: buckets.len(),
            hasher: ah.hasher.clone(),
            max_load_factor: ah.max_load_factor,
            slot_size: buckets[0].len()
        }
    }
}

impl<H> ArrayHashBuilder<H> where H: core::hash::Hasher {
    /// Create new ArrayHashBuilder by using given hasher.
    /// As currently is, the default allocated number of slot per bucket is (4096 / size of usize) slots.
    /// 
    /// Since all slots are Vec, it will be re-allocate if it grow larger than this default.
    #[inline]
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
    #[inline]
    pub fn hasher<H2>(self, hasher: H2) -> ArrayHashBuilder<H2> {
        ArrayHashBuilder {
            hasher,
            buckets_size: self.buckets_size,
            max_load_factor: self.max_load_factor,
            slot_size: self.slot_size
        }
    }

    /// Change buckets size of [ArrayHasher](struct.ArrayHasher.html).
    /// Buckets size scale once max_load_factor is reached.
    /// The new size after it scaled is double of old size.
    /// 
    /// # Parameter
    /// `size` - A number of buckets in this table. It must be greater than 0.
    pub fn buckets_size(mut self, size: usize) -> Self {
        debug_assert!(size > 0);
        self.buckets_size = size;
        self
    }

    /// Change max number of entry before double the buckets size.
    /// 
    /// # Parameter
    /// `factor` - A number of item before it double the bucket size.
    pub fn max_load_factor(mut self, factor: usize) -> Self {
        debug_assert!(factor > 0);
        self.max_load_factor = factor;
        self
    }

    /// Default initialized slot size. Every slot in bucket will be 
    /// allocated by given size. 
    /// Keep in mind that each slot is a `vec`. It can grow pass this
    /// number. It'd be best to give a proper estimation to prevent unnecessary
    /// re-allocation.
    /// 
    /// # Parameter
    /// `size` - A default size for each slot. 
    pub fn slot_size(mut self, size: usize) -> Self {
        self.slot_size = size;
        self
    }

    /// Consume this builder and construct a new [ArrayHash](struct.ArrayHash.html)
    pub fn build<K, V>(self) -> ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq {
        ArrayHash {
            buckets: Some((0..self.buckets_size).map(|_| Vec::with_capacity(self.slot_size)).collect()),
            hasher: self.hasher,
            capacity: self.buckets_size,
            max_load_factor: self.max_load_factor,
            size: 0
        }
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
/// The data can be anything that implement `Hash`. 
/// 
/// The default `Hasher`, if not provided, will be `XxHash64`.
#[derive(Clone, Debug)]
pub struct ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + PartialEq {
    buckets: Option<Box<[Vec<(K, V)>]>>,
    hasher: H,
    capacity: usize,
    max_load_factor: usize,
    size: usize
}

/// Generalize implementation that let two array hash of different type of key and value to be
/// comparable. 
/// It requires that both side must use the same hasher with exactly same seed.
/// It rely on a contract that if any `K1 == K2` then it mean those two hash is also
/// equals.
impl<H, K1, V1, K2, V2> PartialEq<ArrayHash<H, K1, V1>> for ArrayHash<H, K2, V2>
where H: Clone + Hasher + PartialEq,
      K1: Hash + PartialEq + PartialEq<K2>,
      V1: PartialEq<V2>,
      K2: Hash + PartialEq
{
    fn eq(&self, rhs: &ArrayHash<H, K1, V1>) -> bool {
        self.is_hasher_eq(rhs) && 
        self.capacity == rhs.capacity &&
        self.size == rhs.size &&
        rhs.buckets.as_deref().unwrap().iter().zip(self.buckets.as_deref().unwrap().iter()).all(|(rhs, lhs)| {
            rhs.len() == lhs.len() && // Guard case where one side is prefix array of another
            rhs.iter().zip(lhs.iter()).all(|((k1, v1), (k2, v2))| {k1 == k2 && v1 == v2})
        })
    }
}

/// Implement hash by using only `K` and `V` pair to calculate hash.
/// It will still conform to following contract:
/// For any `AH1 ==  AH2`, `hash(AH1) == hash(AH2)`.
/// 
/// This is because the `PartialEq` implementation check if both side have
/// the same size, same hasher, same number of items, and exactly the same
/// key and value pair returned by each yield of both AH1's and AH2's iterator. 
/// However, hasher contract need no symmetric property.
/// It mean that `hash(AH1) == hash(AH2) ` doesn't imply that `AH1 == AH2`.
/// Thus, we will have the same hash if we iterate through array and hash both key and value
/// for each yield on both `AH1` and `AH2`.
impl<H, K, V> Hash for ArrayHash<H, K, V> where H: Hasher + Clone, K: Hash + PartialEq, V: Hash {
    fn hash<H2: Hasher>(&self, hasher: &mut H2) {
        // Since rust iterate on slice to hash and `Box<T>` is hash by deref into `T`,
        // we can simply hash on `self.buckets`. It is equals to iterate on each
        // inner element then hash each individual of it.
        self.buckets.hash(hasher);
    }
}

impl<H, K, V> ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + PartialEq {

    /// Make a builder with default specification equals to current specification of this
    /// array. 
    /// 
    /// Note: The current specification of array may be different from the spec used to create
    /// the array. For example, if original `max_load_factor` is `2` but 3 elements were put
    /// into array, the `max_load_factor` will be `4` and the `bucket_size` will be double of 
    /// the original `bucket_size`.
    #[inline]
    pub fn to_builder(&self) -> ArrayHashBuilder<H> {
        ArrayHashBuilder::from(self)
    }

    /// Check if two array use the same hasher with exactly same seed.
    /// This mean that value `A == B` on these two array will have exactly the same hash value.
    #[inline]
    pub fn is_hasher_eq<K2, V2>(&self, rhs: &ArrayHash<H, K2, V2>) -> bool
    where H: PartialEq,
          K2: Hash + PartialEq
    {
        self.hasher == rhs.hasher
    }

    /// Add or replace entry into this `HashMap`.
    /// If entry is replaced, it will be return in `Option`.
    /// Otherwise, it return `None`
    /// 
    /// # Parameter
    /// - `entry` - A tuple to be add to this.
    /// 
    /// # Return
    /// Option that contains tuple of (key, value) that got replaced or `None` if it is
    /// new entry
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        let mut index = self.make_key(&key);
        let result;

        if let Some(i) = self.buckets.as_mut().unwrap()[index].iter().position(|(k, _)| *k == key) {
            result = Some(self.buckets.as_mut().unwrap()[index].swap_remove(i));
        } else {
            self.size += 1;

            if self.maybe_expand() {
                index = self.make_key(&key);
            }

            result = None
        }

        self.buckets.as_mut().unwrap()[index].push((key, value));

        result
    }

    /// Try to put value into this `ArrayHash`.
    /// If the given key is already in used, leave old entry as is 
    /// and return key/value along with current reference to value associated with the key.
    /// Otherwise, add entry to this `ArrayHash` and return reference to current value.
    /// 
    /// # Parameter
    /// - `entry` - A tuple of (key, value) to be add to this.
    /// # Return
    /// It return `Ok(&V)` if key/value were put into this collection.
    /// Otherwise, it return `Err((K, V, &V))` where `K` is given key,
    /// `V` is given value, and `&V` is reference to current value associated
    /// with given key
    pub fn try_put(&mut self, key: K, value: V) -> Result<&V, (K, V, &V)> {
        let mut index = self.make_key(&key);

        if let Some(i) = self.buckets.as_ref().unwrap()[index].iter().position(|(k, _)| *k == key) {
            Err((key, value, &self.buckets.as_ref().unwrap()[index][i].1))
        } else {
            self.size += 1;
            if self.maybe_expand() {
                index = self.make_key(&key);
            }
            let bucket = &mut self.buckets.as_mut().unwrap()[index];
            bucket.push((key, value));
            Ok(&bucket[bucket.len() - 1].1)
        }
    }

    /// Get a value of given key from this `ArrayHash` relying on contractual
    /// implementation of `PartialEq` and `Hash` where following contract applied:
    /// - for any `A == B` then `B == A` then hash of `A` == hash of `B`
    /// - for any `A == B` and `B == C` then `A == C` then hash of `A` == hash of `B` == hash of `C`
    /// 
    /// # Parameter
    /// - `key` - A key to look for. 
    /// 
    /// # Return
    /// An `Option` contains a value or `None` if it is not found.
    pub fn get<T>(&self, key: &T) -> Option<&V> where T: core::hash::Hash + PartialEq<K> {
        let index = self.make_key(&key);
        let slot = &self.buckets.as_ref().unwrap()[index];
        return slot.iter().find(|(k, _)| {key == k}).map(|(_, v)| v)

        // for (ref k, ref v) in slot.iter() {
        //     if *k == *key {
        //         return Some(v)
        //     }
        // }
        // None
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
    pub fn smart_get<T, Q>(&self, key: Q) -> Option<&V> where Q: core::ops::Deref<Target=T>, K: core::ops::Deref<Target=T>, T: core::hash::Hash + core::cmp::PartialEq {
        let index = self.make_key(&*key);

        let slot = &self.buckets.as_ref().unwrap()[index];
        slot.iter().find(|(k, _)| **k == *key).map(|(_, v)| v)

        // for (ref k, ref v) in slot.iter() {
        //     if **k == *key {
        //         return Some(v)
        //     }
        // }
        // None
    }

    /// Get a value of given key from this `ArrayHash` relying on contractual
    /// implementation of `PartialEq` and `Hash` where following contract applied:
    /// - `B` can be borrowed as type `A`
    /// - for any `A == B` then `B == A` then hash of `A` == hash of `B`
    /// - for any `A == B` and `B == C` then `A == C` then hash of `A` == hash of `B` == hash of `C`
    /// - for any `&A == &B` then `A == B`
    /// 
    /// It is useful for case where the stored key and query key is different type but the stored key
    /// can be borrow into the same type as query. For example, stored `Vec<T>` but query with `&[T]`.
    /// It isn't possible to use [get](struct.ArrayHash.html#method.get) method as `[T]` 
    /// isn't implement `PartialEq<Vec<T>>`.
    /// 
    /// # Parameter
    /// - `key` - A key to look for. 
    /// 
    /// # Return
    /// An `Option` contains a value or `None` if it is not found.
    pub fn coerce_get<T>(&self, key: &T) -> Option<&V> where T: core::hash::Hash + PartialEq + ?Sized, K: Borrow<T> {
        let index = self.make_key(key);
        let slot = &self.buckets.as_ref().unwrap()[index];
        return slot.iter().find(|(k, _)| {key == k.borrow()}).map(|(_, v)| v)

        // for (ref k, ref v) in slot.iter() {
        //     if *k == *key {
        //         return Some(v)
        //     }
        // }
        // None
    }

    /// Attempt to remove entry with given key from this `ArrayHash` relying on contractual
    /// implementation of `PartialEq` and `Hash` where following contract applied:
    /// - for any `A == B` then `B == A` then hash of `A` == hash of `B`
    /// - for any `A == B` and `B == C` then `A == C` then hash of `A` == hash of `B` == hash of `C`
    /// 
    /// # Parameter
    /// - `key` - A key of entry to be remove.
    /// 
    /// # Return
    /// Option that contain tuple of (key, value) or `None` of key is not found
    pub fn remove<T>(&mut self, key: &T) -> Option<(K, V)> where T: core::hash::Hash + PartialEq<K> {
        let slot_idx = self.make_key(key);
        let slot = self.buckets.as_mut().unwrap();
        let entry_idx = slot[slot_idx].iter().position(|(k, _)| {key == k});
        if let Some(i) = entry_idx {
            self.size -= 1;
            Some(slot[slot_idx].swap_remove(i))
        } else {
            None
        }
    }

    /// Attempt to remove entry with given key from this `ArrayHash`.
    /// 
    /// This is usable only if the key is a type of smart pointer that can be deref into another type
    /// which implement `Hash` and `PartialEq`.
    /// 
    /// For example, if K is `Box<[u8]>`, you can use `&[u8]` to remove it.
    /// 
    /// # Parameter
    /// - `key` - A key of entry to be remove.
    /// 
    /// # Return
    /// Option that contain tuple of (key, value) or `None` of key is not found
    pub fn smart_remove<Q, T>(&mut self, key: Q) -> Option<(K, V)>  where Q: core::ops::Deref<Target=T>, K: core::ops::Deref<Target=T>, T: core::hash::Hash + core::cmp::PartialEq {
        let slot_idx = self.make_key(&*key);
        let slot = self.buckets.as_mut().unwrap();
        let entry_idx = slot[slot_idx].iter().position(|(k, _)| {*key == **k});
        if let Some(i) = entry_idx {
            self.size -= 1;
            Some(slot[slot_idx].swap_remove(i))
        } else {
            None
        }
    }

    /// Attempt to remove entry with given key from this `ArrayHash` relying on contractual
    /// implementation of `PartialEq` and `Hash` where following contract applied:
    /// - `B` can be borrowed as type `A`
    /// - for any `A == B` then `B == A` then hash of `A` == hash of `B`
    /// - for any `A == B` and `B == C` then `A == C` then hash of `A` == hash of `B` == hash of `C`
    /// - for any `&A == &B` then `A == B`
    /// 
    /// It is useful for case where the stored key and query key is different type but the stored key
    /// can be borrow into the same type as query. For example, stored `Vec<T>` but query with `&[T]`.
    /// It isn't possible to use [remove](struct.ArrayHash.html#method.remove) method as `[T]` 
    /// isn't implement `PartialEq<Vec<T>>`.
    /// 
    /// # Parameter
    /// - `key` - A key of entry to be remove.
    /// 
    /// # Return
    /// Option that contain tuple of (key, value) or `None` of key is not found
    pub fn coerce_remove<T>(&mut self, key: &T) -> Option<(K, V)> where T: core::hash::Hash + PartialEq + ?Sized, K: Borrow<T> {
        let slot_idx = self.make_key(key);
        let slot = self.buckets.as_mut().unwrap();
        let entry_idx = slot[slot_idx].iter().position(|(k, _)| {key == k.borrow()});
        if let Some(i) = entry_idx {
            self.size -= 1;
            Some(slot[slot_idx].swap_remove(i))
        } else {
            None
        }
    }

    /// Current number of entry in this `ArrayHash`
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if this array hash contains every entry found in given other iterator that yield
    /// `&(K, V)` and `V` implements `PartialEq`.
    /// 
    /// # Parameter
    /// - `other` - A type that implement `IntoIterator<Item=&(K, V)>`
    /// 
    /// # Return
    /// true if this array hash contains every entry that other iterator yield. Otherwise, false.
    pub fn contains_iter<'a, I>(&self, other: I) -> bool where I: IntoIterator<Item=&'a (K, V)>, K: 'a, V: 'a + PartialEq {
        for (key, value) in other.into_iter() {
            if let Some(v) = self.get(key) {
                if v != value {
                    return false
                }
            } else {
                return false
            }
        }

        true
    }

    /// Get an iterator over this `ArrayHash`. 
    /// 
    /// # Return
    /// [ArrayHashIterator](struct.ArrayHashIterator.html) that return reference
    /// to each entry in this `ArrayHash`
    pub fn iter(&self) -> ArrayHashIterator<'_, K, V> {
        let slots = self.buckets.as_ref().unwrap();
        let mut buckets = slots.iter();
        let first_iter = buckets.next().unwrap().iter();
        ArrayHashIterator {
            buckets: buckets,
            current_iterator: first_iter,
            size: self.size
        }
    }

    /// Get a mutable iterator over this `ArrayHash`.
    /// 
    /// Warning, you shall not modify the key part of entry. If you do, it might end up
    /// accessible only by iterator but not with [get](struct.ArrayHash.html#method.get).
    /// 
    /// # Return
    /// [ArrayHashIterMut](struct.ArrayHashIterMut.html) that return mutably reference
    /// to each entry in this `ArrayHash`
    pub fn iter_mut(&mut self) -> ArrayHashIterMut<'_, K, V> {
        if self.size > 0 {
            let buckets: Box<[core::slice::IterMut<(K, V)>]> = self.buckets.as_mut().unwrap().iter_mut().filter_map(|slot| {
                if slot.len() > 0 {Some(slot.iter_mut())} else {None} 
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
                buckets: vec![[].iter_mut()].into_boxed_slice(),
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
    pub fn drain_with<F>(&mut self, pred: F) -> DrainWithIter<F, K, V> where F: Fn(&(K, V)) -> bool {
        let mut bucket_iter = self.buckets.as_mut().unwrap().iter_mut();
        let current_slot = bucket_iter.next();
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
    /// # Parameter
    /// `pred` - A closure that evaluate an entry. If it return true, the entry shall be moved into a new 
    /// [ArrayHash](struct.ArrayHash.html).
    /// 
    /// # Return
    /// An [ArrayHash](struct.ArrayHash.html) that contains all entry that `pred` evaluate to true.
    pub fn split_by<F>(&mut self, pred: F) -> ArrayHash<H, K, V> where F: Fn(&(K, V)) -> bool {
        let mut other = self.to_builder().build();
        let buckets = self.buckets.as_mut().unwrap();
        for i in 0..buckets.len() {
            let mut j = 0;

            loop {
                if j >= buckets[i].len() {
                    break;
                }
                if pred(&buckets[i][j]) {
                    other.buckets.as_mut().unwrap()[i].push(buckets[i].swap_remove(j));
                    other.size += 1;
                    self.size -= 1;
                } else {
                    j += 1;
                }
            }
        }

        other
    }

    /// Since version 0.1.3, any type is acceptable as long as it implements `Hash`.
    #[inline(always)]
    fn make_key<T>(&self, key: &T) -> usize where T: core::hash::Hash + ?Sized {
        let mut local_hasher = self.hasher.clone();
        key.hash(&mut local_hasher);
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

        let old_buckets = self.buckets.take().unwrap().into_vec();
        let new_capacity = self.capacity * 2;
        self.capacity = new_capacity;
        self.max_load_factor *= 2;
        // Assume hash is evenly distribute entry in bucket, the new slot size shall be <= old slot size.
        // This is because the bucket size is doubled.
        let mut buckets: Vec<Vec<(K, V)>> = (0..new_capacity).map(|_| Vec::with_capacity(old_buckets[0].len())).collect();
        old_buckets.into_iter().for_each(|slot| {
            for (key, value) in slot {
                let index = self.make_key(&key);
                buckets[index % new_capacity].push((key, value));
            }
        });
        self.buckets = Some(buckets.into_boxed_slice());

        true
    }
}

/// An iterator that return a reference to each entry in `ArrayHash`.
/// It is useful for scanning entire `ArrayHash`.
#[derive(Debug)]
pub struct ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    buckets: core::slice::Iter<'a, Vec<(K, V)>>,
    current_iterator: core::slice::Iter<'a, (K, V)>,
    size: usize
}

impl<'a, K, V> Iterator for ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    type Item=&'a (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.current_iterator.next();

        while result.is_none() {
            if let Some(slot) = self.buckets.next() {
                self.current_iterator = slot.iter();
                result = self.current_iterator.next();
            } else {
                break
            }
        }

        result
    }
}

impl<'a, K, V> core::iter::FusedIterator for ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {}

impl<'a, K, V> core::iter::ExactSizeIterator for ArrayHashIterator<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    #[inline]
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
pub struct ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    buckets: Box<[core::slice::IterMut<'a, (K, V)>]>,
    remain_slots: usize,
    slot_cursor: usize,
    size: usize
}

impl<'a, K, V> Iterator for ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    type Item=&'a mut (K, V);

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

impl<'a, K, V> core::iter::FusedIterator for ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {}

impl<'a, K, V> core::iter::ExactSizeIterator for ArrayHashIterMut<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    #[inline]
    fn len(&self) -> usize {
        self.size
    }
}

#[derive(Debug)]
pub struct ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    buckets: std::vec::IntoIter<Vec<(K, V)>>,
    current_iterator: std::vec::IntoIter<(K, V)>,
    size: usize
}

impl<K, V> Iterator for ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    type Item=(K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.current_iterator.next();

        while result.is_none() {
            if let Some(slot) = self.buckets.next() {
                if slot.len() > 0 { // skip those slot that have 0 entry
                    self.current_iterator = slot.into_iter();
                    result = self.current_iterator.next();
                }
            } else {
                // entire ArrayHash is exhausted
                break
            }
        }

        result
    }
}

impl<K, V> core::iter::FusedIterator for ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq {}

impl<K, V> core::iter::ExactSizeIterator for ArrayHashIntoIter<K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    #[inline]
    fn len(&self) -> usize {
        self.size
    }
}

impl<H, K, V> IntoIterator for ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq {
    type Item=(K, V);
    type IntoIter=ArrayHashIntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        if self.size >= 1 {
            let mut buckets = self.buckets.unwrap().into_vec().into_iter();
            let current_iterator = buckets.next().unwrap().into_iter();
            ArrayHashIntoIter {
                buckets,
                current_iterator,
                size: self.size
            }
        } else {
            let mut emptied_bucket = vec![vec![]].into_iter();
            let emptied_iterator = emptied_bucket.next().unwrap().into_iter();
            ArrayHashIntoIter {
                buckets: emptied_bucket,
                current_iterator: emptied_iterator,
                size: 0
            }
        }
    }
}

impl<'a, H, K, V> IntoIterator for &'a ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq {
    type Item=&'a(K, V);
    type IntoIter=ArrayHashIterator<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, H, K, V> IntoIterator for &'a mut ArrayHash<H, K, V> where H: core::hash::Hasher + Clone, K: core::hash::Hash + core::cmp::PartialEq {
    type Item=&'a mut (K, V);
    type IntoIter=ArrayHashIterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// An iterator that will drain it underlying [ArrayHash](struct.ArrayHash.html).
#[derive(Debug)]
pub struct DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    bucket_iter: core::slice::IterMut<'a, Vec<(K, V)>>,
    current_slot: Option<&'a mut Vec<(K, V)>>,
    size: usize,
}

impl<'a, K, V> Iterator for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    type Item=(K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.current_slot.as_mut().unwrap().pop();

        while result.is_none() {
            self.current_slot = self.bucket_iter.next();
            if self.current_slot.is_some() {
                result = self.current_slot.as_mut().unwrap().pop();
            } else {
                break;
            }
        }

        result
    }
}

impl<'a, K, V> core::iter::FusedIterator for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {}
impl<'a, K, V> core::iter::ExactSizeIterator for DrainIter<'a, K, V> where K: core::hash::Hash + core::cmp::PartialEq {
    #[inline]
    fn len(&self) -> usize {
        self.size
    }
}

/// An iterator that remove and return element that satisfy predicate.
/// It will also update the size of borrowed [ArrayHash](struct.ArrayHash.html) on each
/// iteration.
#[derive(Debug)]
pub struct DrainWithIter<'a, F, K, V> where F: for<'r> Fn(&'r (K, V)) -> bool, K: core::hash::Hash + core::cmp::PartialEq {
    bucket_iter: core::slice::IterMut<'a, Vec<(K, V)>>,
    cur_size: &'a mut usize,
    current_slot: Option<&'a mut Vec<(K, V)>>,
    predicate: F,
    size: usize
}

impl<'a, F, K, V> Iterator for DrainWithIter<'a, F, K, V> where F: for<'r> Fn(&'r (K, V)) -> bool, K: core::hash::Hash + core::cmp::PartialEq {
    type Item=(K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ref mut v) = self.current_slot {
            for i in 0..v.len() {
                if (self.predicate)(&v[i]) {
                    // Found match
                    *self.cur_size -= 1;
                    return Some(v.swap_remove(i))
                }
            }

            loop {
                self.current_slot = self.bucket_iter.next();
                if self.current_slot.is_some() {
                    if self.current_slot.as_ref().unwrap().len() == 0 {
                        // Keep iterating until non-empty slot is found
                        continue;
                    } else {
                        // Found bucket that has some slot to evaluate
                        break;
                    }
                } else {
                    // All slot in every buckets are evaulated now
                    return None
                }
            }
        }

        None
    }
}

impl<'a, F, K, V> core::iter::FusedIterator for DrainWithIter<'a, F, K, V> where F: for<'r> Fn(&'r (K, V)) -> bool, K: core::hash::Hash + core::cmp::PartialEq {}
impl<'a, F, K, V> core::iter::ExactSizeIterator for DrainWithIter<'a, F, K, V> where F: for<'r> Fn(&'r (K, V)) -> bool, K: core::hash::Hash + core::cmp::PartialEq {
    #[inline]
    fn len(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests;