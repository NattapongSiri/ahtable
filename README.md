# ahtable
An array hash data structure where array is allocated up front with specific size and each element is assigned to each index based on hash function. Each element in an array is a Vec. Each collision of hash will be push into Vec. When number of added element reach certain size, it will scale the size of array by 2x and all old entry will be moved from old array to new one. All of this happen behind the scene. User will not effect by such operation.

# How to use
1. Use `ArrayHashBuilder::default()`
1. Config all required specification of array hash with the builder
1. Use `build()` method on the instance to obtain `ArrayHash`
1. Use following method to operate on the array
    1. `put()` to put new data into array. This method always put or replace existing value.
    1. `try_put()` to put new data into array if it is not already exist.
    1. `get()` to retrieve a value that was put into it.
    1. `smart_get()` to retrieve a value when `key` is of smart pointer type. This will help reduce time/processor required to construct a smart pointer of the same type as key itself.
    1. `remove()` to remove a value out of this array.
    1. `iter()` to iterate over the array entry.
    1. `iter_mut()` to iterate and mutate the array entry. This iterator shall not be used to mutate the entry key.
    1. `drain()` to obtain an iterator that keep draining all data from this array out.
    1. `drain_with()` to obtain an iterator that drain some specific entry out when predicate return true.
    1. `split_by()` which return another array and move all value that satisfy the predicate into new array.
