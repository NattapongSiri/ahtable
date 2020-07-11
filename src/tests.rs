use super::*;

use fnv::FnvHasher;
struct CustomHasher {
    key: u64,
    hasher: FnvHasher
}

impl Clone for CustomHasher {
    fn clone(&self) -> CustomHasher {
        CustomHasher {
            key: self.key,
            hasher: FnvHasher::with_key(self.key)
        }
    }
}

impl core::hash::Hasher for CustomHasher {
    fn write(&mut self, input: &[u8]) {
        self.hasher.write(input);
    }

    fn finish(&self) -> u64 {
        self.hasher.finish()
    }
}

#[test]
fn compare_hash() {
    let mut ah = ArrayHashBuilder::default().build();
    ah.put(1, 1);
    assert_eq!(ah.make_key(&1), ah.make_compound_key(vec![1].iter()));
    assert_ne!(ah.make_key(&1), ah.make_compound_key(vec![1, 2].iter()));
}

#[test]
fn make_array_hash() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }

    let mut counter = 0;
    for _ in ah { // iterate and count number of entry manually
        counter += 1;
    }

    assert_eq!(counter, MAX); 
}

#[test]
fn put_get_xx() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(10_000).build();
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }

    let begin = std::time::Instant::now();
    for j in 0..MAX {
        if let Some(v) = ah.get(&j) {
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }

    dbg!(begin.elapsed().as_millis());
}
#[test]
fn put_get_struct() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(10_000).build();
    #[derive(Clone, Hash, PartialEq)]
    struct Something {
        a: usize
    }
    const MAX :usize = 10_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put_compound(vec![Something {a:i}], i);
    }

    let begin = std::time::Instant::now();
    for j in 0..MAX {
        if let Some(v) = ah.get(&Something{a:j}) {
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }

    dbg!(begin.elapsed().as_millis());
}

#[test]
fn put_smart_get_xx() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(10_000).build();
    
    const MAX :usize = 100_000;
    dbg!("Adding");
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(Box::new(i), i); // Put value using smart pointer Box
    }

    let begin = std::time::Instant::now();
    dbg!("Reading");
    for j in 0..MAX {
        if let Some(v) = ah.smart_get(&j) { // Get value using borrow type
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }
    dbg!("ended");
    dbg!(begin.elapsed().as_millis());
}

#[test]
fn make_custom_hasher() {
    let seed = 0u64;
    let hasher = CustomHasher {
        key: seed,
        hasher: FnvHasher::with_key(seed)
    };
    let mut ah = ArrayHashBuilder::with_hasher(hasher).build();
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }

    let mut counter = 0;

    for _ in ah { // iterate and count number of entry manually
        counter += 1;
    }

    assert_eq!(counter, MAX); 
}


#[test]
fn put_get_fnv() {
    let seed = 0u64;
    let hasher = CustomHasher {
        key: seed,
        hasher: FnvHasher::with_key(seed)
    };
    let mut ah = ArrayHashBuilder::with_hasher(hasher).build();
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }

    for j in 0..MAX {
        if let Some(v) = ah.get(&j) {
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }
}

#[test]
fn test_iter() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }
    let mut ah_vec = ah.into_iter().map(|(mut k, _)| k.next().unwrap()).collect::<Vec<usize>>();
    ah_vec.sort();
    for i in 0..ah_vec.len() {
        assert_eq!(i, ah_vec[i]);
    }
}

#[test]
fn test_iter_mut() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }
    ah.iter_mut().enumerate().for_each(|(i, entry)| {*entry.1 = i;});
    
    for (i, (_, v)) in ah.iter().enumerate() {
        assert_eq!(i, *v);
    }

    let mut sorted: Vec<usize> = ah.into_iter().map(|(mut k, _)| k.next().unwrap()).collect();
    sorted.sort();

    for (i, k) in sorted.into_iter().enumerate() {
        assert_eq!(i, k);
    }
}

#[test]
fn test_drain() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }
    
    let mut keys : Vec<usize> = ah.drain().map(|(mut k, _)| k.next().unwrap()).collect();
    keys.sort_unstable();
    keys.into_iter().enumerate().for_each(|(i, k)| assert_eq!(i, k));
    assert_eq!(ah.len(), 0);
}

#[test]
fn test_drain_with() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(i, i);
    }
    
    let mut keys : Vec<usize> = ah.drain_with(|_, v| {
        *v >= MAX / 2
    }).map(|(_, v)| v).collect();
    let mut count = 0;
    keys.sort_unstable();
    keys.into_iter().enumerate().for_each(|(i, k)| {
        count += 1;
        assert_eq!(i + MAX / 2, k);
    });

    assert_eq!(count, MAX / 2);
    assert_eq!(ah.len(), MAX / 2);
}

#[test]
fn test_split_with() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 hundred thousand entry with one usize for each key and value
        ah.put(i, i);
    }
    
    let ah2 = ah.split_by(|k, _| {
        k[0] >= MAX / 2
    });
    assert_eq!(ah.len(), MAX / 2);
    assert_eq!(ah2.len(), MAX / 2);

    for i in 0..(MAX / 2) {
        if let Some(v) = ah.get(&i) {
            assert_eq!(*v, i);
        } else {
            panic!("Value missing for {}", i)
        }
    }
    for i in (MAX / 2)..MAX {
        if let Some(v) = ah2.get(&i) {
            assert_eq!(*v, i);
        } else {
            panic!("Value missing for {}", i)
        }
    }
}

#[test]
fn test_basic_array_slot() {
    let max = 1_000_000;
    let mut arr: ArraySlot<usize, usize> = ArraySlot::with_capacity(max);

    for i in 0..max {
        arr.push_back(i, i);
    }
    for i in 0..max {
        let (mut key, value) = arr.pop_front().unwrap();
        assert_eq!(key.next().unwrap(), i);
        assert_eq!(value, i);
    }
}

#[test]
fn test_box_array_slot() {
    let mut arr: ArraySlot<Box<usize>, Box<usize>> = ArraySlot::new();

    dbg!("Adding data to array slot");
    for i in 0..100_000 {
        arr.push_back(Box::new(i), Box::new(i));
    }
    dbg!("Popping data out from array slot and check validity");
    for i in 0..100_000 {
        let (mut key, value) = arr.pop_front().unwrap();
        assert_eq!(*(key.next().unwrap()), i);
        assert_eq!(*value, i);
    }
    dbg!("Done");
}

#[test]
fn test_vec_array_slot() {
    let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

    dbg!("Adding data to array slot");
    for i in 0..10_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_back(v.clone(), v);
    }
    dbg!("Popping data out from array slot and check validity");
    for i in 0..10_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        let (mut key, value) = arr.pop_front().unwrap();
        assert_eq!(key.next().unwrap(), v);
        assert_eq!(value, v);
    }
    dbg!("Done");
}

#[test]
fn test_compound_vec_array_slot() {
    let mut arr: ArraySlot<usize, Vec<usize>> = ArraySlot::new();

    dbg!("Adding data to array slot");
    for i in 0..10_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        let keys = v.clone().into_iter();
        arr.push_compound_back(keys, v);
    }
    dbg!("Popping data out from array slot and check validity");
    for i in 0..10_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        let (key, value) = arr.pop_front().unwrap();
        assert_eq!(key.collect::<Vec<usize>>(), v);
        assert_eq!(value, v);
    }
    dbg!("Done");
}
#[test]
fn test_compound_array_slot_into_iter() {
    let mut arr: ArraySlot<usize, Vec<usize>> = ArraySlot::new();
    let max = 100_000;
    for i in 0..max {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_compound_back(v.clone().into_iter(), v);
    }
    let mut count = 0;
    for (i, (keys, value)) in arr.into_iter().enumerate() {
        let mut key_len = 0;
        keys.zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
            assert_eq!(i + j, k);
            assert_eq!(i + j, *v);
            key_len += 1;
        });
        assert_eq!(key_len, 100);
        count += 1;
    }
    assert_eq!(count, max);
}

#[test]
fn test_compound_array_slot_iter() {
    let mut arr: ArraySlot<usize, Vec<usize>> = ArraySlot::new();

    for i in 0..100_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_compound_back(v.clone().into_iter(), v);
    }
    let mut count = 0;
    for (i, (keys, value)) in arr.iter().expect("Unexpected error return from getting iter on contiguous ArraySlot").enumerate() {
        let mut key_len = 0;
        keys.zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
            assert_eq!(i + j, *k);
            assert_eq!(i + j, *v);
            key_len += 1;
        });
        assert_eq!(key_len, 100);
        count += 1;
    }

    assert_eq!(count, 100_000);
}
#[test]
fn test_vec_array_slot_iter() {
    let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

    for i in 0..100_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_back(v.clone(), v);
    }
    let mut count = 0;
    for (i, (mut keys, value)) in arr.iter().expect("Unexpected error return from getting iter on contiguous ArraySlot").enumerate() {
        let mut key_len = 0;
        keys.next().unwrap().iter().zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
            assert_eq!(i + j, *k);
            assert_eq!(i + j, *v);
            key_len += 1;
        });
        assert_eq!(key_len, 100);
        count += 1;
    }
    assert_eq!(count, 100_000);
}

#[test]
fn test_compound_array_slot_iter_mut() {
    let mut arr: ArraySlot<usize, Vec<usize>> = ArraySlot::new();

    for i in 0..100_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_compound_back(v.clone().into_iter(), v);
    }
    for (keys, value) in arr.iter_mut().expect("Unexpected error return from getting iter on contiguous ArraySlot") {
        keys.zip(value.iter_mut()).for_each(|(k, v)| {
            *k += 1;
            *v += 2;
        });
    }
    let mut count = 0;
    for (i, (keys, value)) in arr.iter().expect("Unexpected error return from getting iter on contiguous ArraySlot").enumerate() {
        let mut key_len = 0;
        keys.zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
            assert_eq!(i + j + 1, *k);
            assert_eq!(i + j + 2, *v);
            key_len += 1;
        });
        assert_eq!(key_len, 100);
        count += 1;
    }
    assert_eq!(count, 100_000);
}
#[test]
fn test_vec_array_slot_iter_mut() {
    let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

    for i in 0..100_000 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_back(v.clone(), v);
    }
    for (mut keys, value) in arr.iter_mut().expect("Unexpected error return from getting iter on contiguous ArraySlot") {
        keys.next().unwrap().iter_mut().zip(value.iter_mut()).for_each(|(k, v)| {
            *k += 1;
            *v += 2;
        });
    }
    let mut count = 0;
    for (i, (mut keys, value)) in arr.iter_mut().expect("Unexpected error return from getting iter on contiguous ArraySlot").enumerate() {
        let mut key_len = 0;
        keys.next().unwrap().iter().zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
            assert_eq!(i + j + 1, *k);
            assert_eq!(i + j + 2, *v);
            key_len += 1;
        });
        assert_eq!(key_len, 100);
        count += 1;
    }
    assert_eq!(count, 100_000);
}
#[test]
fn test_array_slot_lifetime() {
    fn get_ref<T, U>(arr: &ArraySlot<T, U>) -> (RefKeyIterator<T>, &U) {
        arr.iter().unwrap()
            .next().unwrap()
    }
    let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

    for i in 0..10 {
        let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
        arr.push_back(v.clone(), v);
    }
    get_ref(&arr).0.next().unwrap().iter().zip(0..100).for_each(|(k, v)| {
        assert_eq!(*k, v);
    });
}

#[test]
fn test_vec_array_slot_ownership() {
    fn build_array_slot() -> ArraySlot<Vec<usize>, Vec<usize>> {
        let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

        for i in 0..100_000 {
            let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
            arr.push_back(v.clone(), v);
        }
        arr
    }
    let arr = build_array_slot();
    let mut count = 0;
    for (i, (mut keys, value)) in arr.iter().expect("Unexpected error return from getting iter on contiguous ArraySlot").enumerate() {
        let mut key_len = 0;
        keys.next().unwrap().iter().zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
            assert_eq!(i + j, *k);
            assert_eq!(i + j, *v);
            key_len += 1;
        });
        assert_eq!(key_len, 100);
        count += 1;
    }
    assert_eq!(count, 100_000);
}

#[test]
fn test_array_slot_replace() {
    let mut slot = ArraySlot::new();
    for i in 0..1000 {
        slot.push_back(i, i);
    }
    assert_eq!(slot.replace(&0, 1).unwrap(), 0);
    assert_eq!(slot.replace(&9999, 1).unwrap_err(), 1);
    assert_eq!(slot.replace(&999, 1).unwrap(), 999);
    assert_eq!(slot.replace(&500, 0).unwrap(), 500);
    let mut first = slot.pop_front().unwrap();
    assert_eq!(first.0.next().unwrap(), 0);
    assert_eq!(first.1, 1);
    for i in 1..500 {
        let mut entry = slot.pop_front().unwrap();
        assert_eq!(entry.0.next().unwrap(), i);
        assert_eq!(entry.1, i);
    }
    let mut middle = slot.pop_front().unwrap();
    assert_eq!(middle.0.next().unwrap(), 500);
    assert_eq!(middle.1, 0);
    for i in 501..999 {
        let mut entry = slot.pop_front().unwrap();
        assert_eq!(entry.0.next().unwrap(), i);
        assert_eq!(entry.1, i);
    }
    let mut last = slot.pop_front().unwrap();
    assert_eq!(last.0.next().unwrap(), 999);
    assert_eq!(last.1, 1);
}

#[test]
fn test_array_slot_replace_compound() {
    let mut slot = ArraySlot::new();
    for i in 0..1000 {
        let keys : Vec<u32> = (i..(i + 10)).collect();
        slot.push_compound_back(keys.into_iter(), i);
    }
    // these replace is partial key match, it'd return all err
    assert_eq!(slot.replace(&0, 1).unwrap_err(), 1); 
    assert_eq!(slot.replace(&9999, 1).unwrap_err(), 1);
    assert_eq!(slot.replace(&999, 1).unwrap_err(), 1);
    assert_eq!(slot.replace(&500, 0).unwrap_err(), 0);
    // One of the replace is mismatch, it'd return one err
    assert_eq!(slot.replace_compound((0..10).collect::<Vec<u32>>().as_slice(), 1).unwrap(), 0); 
    assert_eq!(slot.replace_compound((9999..10009).collect::<Vec<u32>>().as_slice(), 1).unwrap_err(), 1);
    assert_eq!(slot.replace_compound((999..1009).collect::<Vec<u32>>().as_slice(), 1).unwrap(), 999);
    assert_eq!(slot.replace_compound((500..510).collect::<Vec<u32>>().as_slice(), 0).unwrap(), 500);
    let first = slot.pop_front().unwrap();
    assert_eq!(first.0.collect::<Vec<u32>>(), (0..10).collect::<Vec<u32>>());
    assert_eq!(first.1, 1);
    for i in 1..500 {
        let entry = slot.pop_front().unwrap();
        assert_eq!(entry.0.collect::<Vec<u32>>(), (i..(i + 10)).collect::<Vec<u32>>());
        assert_eq!(entry.1, i);
    }
    let middle = slot.pop_front().unwrap();
    assert_eq!(middle.0.collect::<Vec<u32>>(), (500..510).collect::<Vec<u32>>());
    assert_eq!(middle.1, 0);
    for i in 501..999 {
        let entry = slot.pop_front().unwrap();
        assert_eq!(entry.0.collect::<Vec<u32>>(), (i..(i + 10)).collect::<Vec<u32>>());
        assert_eq!(entry.1, i);
    }
    let last = slot.pop_front().unwrap();
    assert_eq!(last.0.collect::<Vec<u32>>(), (999..1009).collect::<Vec<u32>>());
    assert_eq!(last.1, 1);
}

#[test]
fn test_array_slot_drain_with() {
    let mut slot = ArraySlot::with_capacity(100);
    for i in 0..100 {
        slot.push_back(i, i);
    }

    let mut count = 0;

    slot.drain_with(|k, _| {k[0] >= 50}).unwrap().enumerate().for_each(|(i, (_, v))| {
        assert_eq!(i + 50, v);
        count += 1;
    });

    assert_eq!(count, 50);
    slot.iter().unwrap().enumerate().for_each(|(i, (mut k, v))| {
        assert_eq!(i, *k.next().unwrap());
        assert_eq!(i, *v);
    });

    slot.drain_with(|_, _| true).unwrap().enumerate().for_each(|(i, (_, v))| {
        assert_eq!(i, v);
        count += 1;
    });

    assert_eq!(count, 100);
}

#[test]
fn test_array_slot_non_contiguous_drain_with() {
    let mut slot = ArraySlot::with_capacity(1_000);
    let max = 100_000;
    // make head point at half of buffer
    for i in 0..(max / 2) {
        slot.push_back(i, i);
        slot.pop_front().unwrap();
    }

    // first 51 elements are on tail
    for i in 0..max {
        slot.push_back(i, i);
    }

    let mut count = 0;
    // drain all elements after 50 out. One of it will be on tail of buffer.
    slot.drain_with(|k, _| {k[0] >= max / 2}).unwrap().enumerate().for_each(|(i, (_, v))| {
        assert_eq!(i + (max / 2), v);
        count += 1;
    });

    assert_eq!(count, max / 2);
    slot.iter().unwrap().enumerate().for_each(|(i, (mut k, v))| {
        assert_eq!(i, *k.next().unwrap());
        assert_eq!(i, *v);
    });

    slot.drain_with(|_, _| true).unwrap().enumerate().for_each(|(i, (_, v))| {
        assert_eq!(i, v);
        count += 1;
    });

    assert_eq!(count, max);
}

/// This test take a long time to complete. This is to let tester observe memory
/// consumption behavior which shall be swing within specific period, e.g. 70-160 MB.
/// This is because in each iteration, it'll build new ArraySlot and drop it when the
/// iteration is completed. If there's no memory leak, it shall peak to the same 
/// amount on every iteration.
/// 
/// If it keep consume more memory then there's a memory leak problem.
/// 
/// By default, this test case will be skipped.
#[test]
#[ignore]
fn test_vec_array_slot_memory_leak() {
    fn build_array_slot() -> ArraySlot<Vec<usize>, Vec<usize>> {
        let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

        for i in 0..100_000 {
            let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
            arr.push_back(v.clone(), v);
        }
        arr
    }
    for _ in 0..1000 {
        let arr = build_array_slot();
        let mut count = 0;
        for (i, (mut keys, value)) in arr.iter().expect("Unexpected error return from getting iter on contiguous ArraySlot").enumerate() {
            let mut key_len = 0;
            keys.next().unwrap().iter().zip(value.iter()).enumerate().for_each(|(j, (k, v))| {
                assert_eq!(i + j, *k);
                assert_eq!(i + j, *v);
                key_len += 1;
            });
            assert_eq!(key_len, 100);
            count += 1;
        }
        assert_eq!(count, 100_000);
    }
}
// Uncomment test case below and it should give a compile error by borrow checker on mutable and immutable borrow
// #[test]
// fn test_array_slot_borrow_checker() {
//     let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

//     for i in 0..100_000 {
//         let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
//         arr.push_back(v.clone(), v);
//     }
//     arr.iter_mut().unwrap().for_each(|_| {
//         arr.iter().unwrap().for_each(|_| {});
//     });
// }

// Uncomment test case below and it should give a compile error by borrow checker on mutable and immutable borrow
// #[test]
// fn test_array_slot_borrow_checker() {
//     let mut arr: ArraySlot<Vec<usize>, Vec<usize>> = ArraySlot::new();

//     for i in 0..100_000 {
//         let v = (0..100).map(|j| {j + i}).collect::<Vec<usize>>();
//         arr.push_back(v.clone(), v);
//     }
//     let a = arr.iter_mut().unwrap().find(|_| {
//         true
//     });
//     let b = arr.iter().unwrap().find(|_| {true});
//     a.unwrap().1[1] = 1;
// }
