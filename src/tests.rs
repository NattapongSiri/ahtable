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
    
    const MAX :usize = 1_000_000;
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
fn put_smart_get_xx() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(10_000).build();
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(Box::new(i), i); // Put value using smart pointer Box
    }

    let begin = std::time::Instant::now();

    for j in 0..MAX {
        if let Some(v) = ah.smart_get(&j) { // Get value using borrow type
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }

    dbg!(begin.elapsed().as_millis());
}

#[test]
fn put_contractual_get_xx() {
    let mut ah = ArrayHashBuilder::default().build();

    #[derive(Hash, PartialEq)]
    struct A(usize);

    #[derive(Hash)]
    struct B(usize);

    impl PartialEq<A> for B {
        fn eq(&self, rhs: &A) -> bool {
            self.0 == rhs.0
        }
    }
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(A(i), i); // Put value using smart pointer Box
    }

    assert!(ah.get(&A(MAX + 1)).is_none());
    let begin = std::time::Instant::now();

    for j in 0..MAX {
        if let Some(v) = ah.get(&B(j)) { // Get value using borrow type
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }

    dbg!(begin.elapsed().as_millis());
}

#[test]
fn put_coalesced_get_xx() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(10_000).build();
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(vec![i], i); // Put value using smart pointer Box
    }

    assert!(ah.coerce_get::<[usize]>(&[MAX + 1]).is_none());
    let begin = std::time::Instant::now();

    for j in 0..MAX {
        if let Some(v) = ah.coerce_get::<[usize]>(&[j]) { // Get value using borrow type
            assert_eq!(*v, j);
        } else {
            panic!("Cannot retrieve value back using existing key")
        }
    }

    dbg!(begin.elapsed().as_millis());
}

#[test]
fn contains_all() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(10_000).build();
    let mut another = ah.clone();
    let mut partly_eq = ah.clone();
    
    const MAX :usize = 1_000_000;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(vec![i], i); // Put value using smart pointer Box
    }

    for i in 0..(MAX / 2) {
        another.put(vec![i], i);
        partly_eq.put(vec![i], 0);
    }

    let begin = std::time::Instant::now();
    assert!(ah.contains_iter(&another));
    assert!(!ah.contains_iter(&partly_eq));
    dbg!(begin.elapsed().as_millis());
}

#[test]
fn partial_eq() {
    let mut ah = ArrayHashBuilder::default().max_load_factor(8).buckets_size(8).build();
    let mut ah_2 = ArrayHashBuilder::default().max_load_factor(8).buckets_size(8).build();
    
    const MAX :usize = 16;
    for i in 0..MAX { // put 1 million entry with one usize for each key and value
        ah.put(vec![i], i); 
        ah_2.put(vec![i], i); // Put similar key/value to see if it's eq
    }

    assert_ne!(ah, ah_2);

    let ah_3 = ah.clone();
    assert_eq!(ah, ah_3);
    let mut ah_4 = ah.to_builder().build();
    let mut ah_5 = ah.to_builder().build();

    for i in 0..MAX {
        ah_4.put(vec![i], i); // Put similar key/value to see if it's eq
        ah_5.put(vec![i], 0); // Put similar key but different value to see if it's eq
    }

    assert_eq!(ah, ah_4);
    assert_ne!(ah, ah_5);
}
#[test]
fn hasher_eq() {
    let ah = ArrayHashBuilder::default().max_load_factor(8).buckets_size(8).build::<u8, u8>();
    let ah_2 = ArrayHashBuilder::default().max_load_factor(8).buckets_size(8).build::<u8, u8>();
    assert!(!ah.is_hasher_eq(&ah_2));
    let ah_3 = ah.clone();
    assert!(ah.is_hasher_eq(&ah_3));
    let ah_4 = ah.to_builder().build::<u8, u8>();
    let ah_5 = ah.to_builder().build::<u8, u8>();
    assert!(ah.is_hasher_eq(&ah_4));
    assert!(ah_5.is_hasher_eq(&ah_4));
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
    let mut ah_vec = ah.iter().map(|(k, _)| *k).collect::<Vec<usize>>();
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
    ah.iter_mut().enumerate().for_each(|(i, entry)| {entry.1 = i;});
    
    for (i, (_, v)) in ah.iter().enumerate() {
        assert_eq!(i, *v);
    }

    let mut sorted: Vec<usize> = ah.into_iter().map(|(k, _)| k).collect();
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
    
    let mut keys : Vec<usize> = ah.drain().map(|(k, _)| k).collect();
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
    
    let mut keys : Vec<usize> = ah.drain_with(|(_, v)| {
        *v >= MAX / 2
    }).map(|(k, _)| k).collect();
    keys.sort_unstable();
    keys.into_iter().enumerate().for_each(|(i, k)| assert_eq!(i + MAX / 2, k));
    assert_eq!(ah.len(), MAX / 2);
}

#[test]
fn test_split_with() {
    let mut ah = ArrayHashBuilder::default().build();
    
    const MAX :usize = 100_000;
    for i in 0..MAX { // put 1 hundred thousand entry with one usize for each key and value
        ah.put(i, i);
    }
    
    let ah2 = ah.split_by(|(k, _)| {
        *k >= MAX / 2
    });
    assert_eq!(ah.len(), 50_000);
    assert_eq!(ah2.len(), 50_000);

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
#[ignore]
fn bench_compound_key() {
    let start = std::time::Instant::now();
    let mut ah = ArrayHashBuilder::default().buckets_size(4096).build();
    for i in 0..100_000 {
        let v: Vec<usize> = (i..(i + 100)).collect();
        ah.put(v.clone(), v);
    }
    let created = start.elapsed().as_millis();
    println!("Created in {}s {}ms", created / 1000, created % 1000);
    for i in 0..100_000 {
        let v: Vec<usize> = (i..(i + 100)).collect();
        if let Some(value) = ah.get(&v) {
            assert_eq!(*value, v);
        }
    }
    let created = start.elapsed().as_millis();
    println!("Got in {}s {}ms", created / 1000, created);
}