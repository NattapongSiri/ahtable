# ahtable
An array hash data structure where array is allocated up front with specific size and each element is assigned to each index based on hash function. Each element in an array is a Vec. Each collision of hash will be push into Vec. When number of added element reach certain size, it will scale the size of array by 2x and all old entry will be moved from old array to new one. All of this happen behind the scene. User will not effect by such operation.

# How to use
1. Use `ArrayHashBuilder::default()` 
1. Config all required specification of array hash with the builder
1. Use `build()` method on the instance to obtain `ArrayHash`
1. Use following method to operate on the array
    1. `put()` to put new data into array. This method always put or replace existing value.
    1. `try_put()` to put new data into array if it is not already exist.
    1. `get()` to retrieve a value that was put into it by a key of a type that is hashable and comparable to key.
    1. `smart_get()` to retrieve a value when `key` is of smart pointer type. This will help reduce time processor required to construct a smart pointer of the same type as key itself.
    1. `coerce_get()` to retrieve a value when `key` is of a type that can be borrowed as another type which isn't implement `PartialEq<K>` to the original type. 
    1. `remove()` to remove a value out of this array by a key of a type that is hashable and comparable to key.
    1. `smart_remove()` to retrieve a value when `key` is of smart pointer type. This will help reduce time processor required to construct a smart pointer of the same type as key itself.
    1. `coerce_remove()` to remove a value when `key` is of a type that can be borrowed as another type which isn't implement `PartialEq<K>` to the original type. 
    1. `contains_iter` to test whether current array have every entry that the iterator yield.
    1. `to_builder` to obtain `ArrayHashBuilder` with current spec out of existing array.
    1. `iter()` to iterate over the array entry.
    1. `iter_mut()` to iterate and mutate the array entry. This iterator shall not be used to mutate the entry key.
    1. `drain()` to obtain an iterator that keep draining all data from this array out.
    1. `drain_with()` to obtain an iterator that drain some specific entry out when predicate return true.
    1. `split_by()` which return another array and move all value that satisfy the predicate into new array.
    1. `is_hasher_eq()` to check whether two array have equivalence hasher.

## Breaking change from 0.1 to 0.2
- `ArrayHash::try_put` is now return `Result`. When it fail to put, it return `Err` with given key/value along with reference to current value associated with given key. When it succeed, it return `Ok` with reference to value that was put into array.

## Migration guide from 0.1 to 0.2
1. for any `if let Some(v) = array_hash.try_put(key, value)`, it'd become `if let Err((key, value, cur_val) = array_hash.try_put(key, value)`
1. for any `if array_hash.try_put(key, value).is_some()`, it'd become `if array_hash.try_put(key, value).is_err()`
1. for any `if array_hash.try_put(key, value).is_none()`, it'd become `if array_hash.try_put(key, value).is_ok()`
1. for any statement `array_hash.try_put(key, value);`, it'd become `array_hash.try_put(key, value).unwrap()`
1. for any `let cur_v = array_hash.try_put(key, value).unwrap()`, it'd become `let (_, _, cur_v) = array_hash.try_put(key, value).unwrap_err()`

## Important notes
1. `PartialEq` of `ArrayHash` need both comparator and comparatee to be exactly the same. This including `Hasher` which must be seed by exactly the same number. The ideal usage is to fill largest array first then use `ArrayHash::to_builder` to build second array. If it is impossible, consider construct an `ArrayHash` that is large enough to stored everything without reaching `max_load_factor` then use `ArrayHash::to_builder` or clone that `ArrayHashBuilder` to build every array.
1. There's method `ArrayHash::is_hasher_eq` to test whether two array can be compared. If two arrays are using different type of hasher, it will immediately yield compile error. If it use the same type of hasher but it use different seed, it will return `false`. Otherwise, it is comparable via `==` operator.
1. Directly using `==` operator on two arrays are safe. It will compile error similar to `ArrayHash::is_hasher_eq`. In fact, in `PartialEq` implementation, it use `ArrayHash::is_hasher_eq` to check first if it is comparable. However, it will always return false if two array use different seed even if both array have exactly the same elements in it.
1. It is much faster to use `==` operator to compare two arrays than using `ArrayHash::contains_iter` as `contains_iter` will need to hash every key return by iter. The `contains_iter` method is suitable in case where two arrays are using different hasher type or built from different ``ArrayHashBuilder`.

# What's new
## 0.2.0
- `ArrayHash::try_put` moved given `key` and `value` but doesn't guarantee to put the `key` and `value` in.
It now return `Result`. If the `key` and `value` is successfully put, it return `Ok((&V))` where `&V` is the reference to value that was put. If it fail to put, it return `Err((K, V))` where `K` is the given `key` and `V` is given value.
## 0.1.5
- Fix hash defect. When hashing on `ArrayHash` itself which may cause two hash to be different eventhough, `==` of two array is true. This is because `PartialEq` doesn't compare `max_load_factor` but in `0.1.4` hash take `max_load_factor` to calculate the hash.
## 0.1.4
- `ArrayHash` and `ArrayHashBuilder` are now implements `Hash` and `PartialEq`
- `ArrayHash::is_hasher_eq` to check if two array use exactly the same hasher.
- `ArrayHash::coerce_get` and `ArrayHash::coerce_remove` that accept a borrowed type that doesn't implement `PartialEq<K>` with the stored entry
- `ArrayHash::smart_remove` which is counterpart of `ArrayHash::smart_get` that is usable when both stored `key` and query can be deref into the same type.
- impl `core::convert::From<ArrayHash>`  for `ArrayHashBuilder`.
- `ArrayHash::to_builder` to retrieve a `ArrayHashBuilder` that can build an `ArrayHash` with exactly same spec as current `ArrayHash`.
- `ArrayHash::contains_iter` that check if this array contain every entry that given iter yield.
## 0.1.3
- `ArrayHash::get` and `ArrayHash::remove` parameter is now generic instead of fixing it to be the same type as the one being key. Now any types that implements `PartialEq<K>` + `Hash` can be used as parameter.
- `ArrayHash` key and value no longer need to implement clone. The reason behind this is because there are two place where it need to clone key and value. Both of it is for purpose of allocating `Vec`. However, in both place, it need no actual clone on key nor value. It allocate an empty `Vec`. Therefore, cloning empty `Vec` would have no different impact on performance comparing to looping to construct an empty `Vec`. With this reason, it would be easier for library user to have lesser number of constraint on key/value.