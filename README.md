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
    1. `remove()` to remove a value out of this array by a key of a type that is hashable and comparable to key.
    1. `iter()` to iterate over the array entry.
    1. `iter_mut()` to iterate and mutate the array entry. This iterator shall not be used to mutate the entry key.
    1. `drain()` to obtain an iterator that keep draining all data from this array out.
    1. `drain_with()` to obtain an iterator that drain some specific entry out when predicate return true.
    1. `split_by()` which return another array and move all value that satisfy the predicate into new array.

# What's new
## 0.1.3
- `ArrayHash::get` and `ArrayHash::remove` parameter is now generic instead of fixing it to be the same type as the one being key. Now any types that implements `PartialEq<K>` + `Hash` can be used as parameter.
- `ArrayHash` key and value no longer need to implement clone. The reason behind this is because there are two place where it need to clone key and value. Both of it is for purpose of allocating `Vec`. However, in both place, it need no actual clone on key nor value. It allocate an empty `Vec`. Therefore, cloning empty `Vec` would have no different impact on performance comparing to looping to construct an empty `Vec`. With this reason, it would be easier for library user to have lesser number of constraint on key/value.