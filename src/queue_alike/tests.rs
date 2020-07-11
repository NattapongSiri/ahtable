use super::*;

#[test]
fn test_basic() {
    let mut queue = QueueAlike::with_capacity(1);
    queue.enqueue(1);
    assert_eq!(queue.dequeue().unwrap(), 1);
}
#[test]
fn test_circulate() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..10 {
        queue.enqueue(i);
        assert_eq!(queue.dequeue().unwrap(), i);
    }

    assert_eq!(queue.capacity(), 1);
}
#[test]
fn test_expand() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..10 {
        queue.enqueue(i);
    }

    assert_eq!(queue.capacity(), 16);
}
#[test]
fn test_circulate_expand() {
    let mut queue = QueueAlike::with_capacity(4);
    for i in 0..4 {
        queue.enqueue(i); // queue is 0, 1, 2, 3
    }
    
    assert_eq!(queue.len(), 4);
    assert_eq!(queue.capacity(), 4);

    for i in 0..2 {
        assert_eq!(queue.dequeue().unwrap(), i); // queue is 2, 3
    }

    assert_eq!(queue.len(), 2);
    assert_eq!(queue.capacity(), 4);

    for i in 4..8 {
        queue.enqueue(i); // queue is 2, 3, 4, 5, 6, 7, 8
    }

    assert_eq!(queue.len(), 6);
    assert_eq!(queue.capacity(), 8);

    for i in 2..8 {
        assert_eq!(queue.dequeue().unwrap(), i); // queue is 2, 3, 4, 5, 6, 7, 8
    }

    assert_eq!(queue.len(), 0);
    assert_eq!(queue.capacity(), 8);
}
#[test]
fn test_split() {
    let mut queue = QueueAlike::with_capacity(10_000);
    for times in 1..=1_000 {
        for i in times..(times + 5_000) {
            queue.enqueue(i);
        }
        if times > 1 {
            let mut other_half = queue.split_off(5_000).unwrap();
            for i in times..(times + 5_000) {
                assert_eq!(other_half.dequeue().unwrap(), i);
            }
        }
    }
}
#[test]
fn test_iter() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }

    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(*v, i)});
}
#[test]
fn test_iter_mut() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    queue.iter_mut().for_each(|v| {*v += 1});
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(*v, i + 1)});
}
#[test]
fn test_into_iter() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    for (i, v) in queue.into_iter().enumerate() {
        assert_eq!(v, i);
    }
}
#[test]
fn test_subqueue_basic_full_subset() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    let subset = queue.subset(5_000, 10_000);
    for (i, v) in subset.into_iter().enumerate() {
        assert_eq!(v - 5_000, i);
    }

    // Uncomment below and it shall compile error because it try to borrow mut twice on same queue

    // let mut mut1 = queue.subset_mut_from(10);
    // let mut2 = queue.subset_mut_to(10);
    // mut1[0] = 1;
}

#[test]
fn test_subqueue_basic_from_subset() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    let subset = queue.subset_from(500_000);
    for (i, v) in subset.into_iter().enumerate() {
        assert_eq!(v - 500_000, i);
    }

    // Uncomment below and it shall compile error because it try to both borrow and borrow mut at the same time

    // let mut mut1 = queue.subset_mut_from(10);
    // let mut2 = queue.subset_to(10);
    // mut1[0] = 1;
}

#[test]
fn test_subqueue_basic_to_subset() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    let subset = queue.subset_to(500_000);
    for (i, v) in subset.into_iter().enumerate() {
        assert_eq!(*v, i);
    }
}

#[test]
fn test_subqueue_tail_head_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-1,000,000
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for i in 0..500_000 {
        assert_eq!(queue.dequeue().unwrap(), i);
    }
    // re-fill the queue again so it's 500,000 - 1,000,000 then 0-500,000
    for i in 0..500_000 {
        queue.enqueue(i);
    }
    // Get last 500,000 subset from queue
    let subset = queue.subset(500_000, 1_000_000);
    for (i, v) in subset.into_iter().enumerate() {
        assert_eq!(*v, i); // it shall be 0-500,000
    }
}

#[test]
fn test_subqueue_tail_head_to_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-1,000,000
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for _ in 0..500_000 {
        queue.dequeue();
    }
    // re-fill the queue again so it's 500,000 - 1,000,000 then 0-500,000
    for i in 0..500_000 {
        queue.enqueue(i);
    }
    // Get first 500,000 subset from queue
    let subset = queue.subset_to(500_000);
    for (i, v) in subset.into_iter().enumerate() {
        assert_eq!(v - 500_000, i); // it shall be 500,000 - 1,000,000
    }
}

#[test]
fn test_subqueue_tail_head_from_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-1,000,000
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for _ in 0..500_000 {
        queue.dequeue();
    }
    // re-fill the queue again so it's 500,000 - 1,000,000 then 0-500,000
    for i in 0..500_000 {
        queue.enqueue(i);
    }
    // Get first 500,000 subset from queue
    let subset = queue.subset_from(500_000);
    for (i, v) in subset.into_iter().enumerate() {
        assert_eq!(*v, i); // it shall be 0 - 500,000
    }
}

#[test]
fn test_subqueue_mut_basic_full_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // use subset_mut and turn second half into 0-500,000
    let mut subset = queue.subset_mut(500_000, 1_000_000);
    for v in subset.iter_mut() { 
        *v -= 500_000; 
    }

    for (i, v) in queue.into_iter().enumerate() {
        assert_eq!(i % 500_000, v);
    }
    // Uncomment statement below and it should give a compile error due to
    // simultaneously having non-mut and mut borrow at the same time
    // subset[0] = 0;
}

#[test]
fn test_subqueue_mut_basic_from_subset() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    let mut subset = queue.subset_mut_from(500_000);
    subset.iter_mut().for_each(|v| {
        *v -= 500_000;
    });

    queue.into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i % 500_000, v);
    });
}

#[test]
fn test_subqueue_mut_basic_to_subset() {
    let mut queue = QueueAlike::with_capacity(1);
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    let mut subset = queue.subset_mut_to(500_000);
    for v in subset.iter_mut() {
        *v += 500_000; // Now we have two halves of 500,000-999,999
    }
    queue.iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i % 500_000 + 500_000, *v);
    });
}

#[test]
fn test_subqueue_mut_tail_head_full() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for i in 0..500_000 {
        assert_eq!(queue.dequeue().unwrap(), i);
    }
    // re-fill the queue again so it's 500,000 - 999,999 then 0-499,999
    for i in 0..500_000 {
        queue.enqueue(i);
    }
    // Get last 500,000 subset from queue
    let mut subset = queue.subset_mut(500_000, 1_000_000);
    for v in subset.iter_mut() {
        *v += 1_000_000 // it was 0-499,999. Now it is 1,000,000 - 1,499,999
    }
    // Now the queue will be 500,000 - 1,499,999
    queue.iter_mut().for_each(|v| {
        *v -= 500_000
    }); // it become two halves of 0-999,999

    queue.into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i, v);
    });

    // Uncomment line below and it should give compile error
    // subset[0] = 0;
}

#[test]
fn test_subqueue_mut_tail_head_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for i in 0..500_000 {
        assert_eq!(queue.dequeue().unwrap(), i);
    }
    // re-fill the queue again so it's 500,000 - 999,999 then another 500,000-999,999
    for i in 500_000..1_000_000 {
        queue.enqueue(i);
    }
    // Get first 500,000 subset from queue then subtract it with 500,000
    let mut subset = queue.subset_mut_to(500_000);
    for v in subset.iter_mut() {
        *v -= 500_000
    }
    // Now the queue will be 0 - 999,999

    queue.subset_from(10_000).into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i + 10_000, *v);
    });

    // Uncomment line below and it should give compile error
    // subset[0] = 0;
}

#[test]
fn test_subqueue_mut_tail_head_to_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for i in 0..500_000 {
        assert_eq!(queue.dequeue().unwrap(), i);
    }
    // re-fill the queue again so it's 500,000 - 999,999 then 0-499,999
    for i in 0..500_000 {
        queue.enqueue(i);
    }
    // Get last 500,000 subset from queue
    let mut subset = queue.subset_mut_from(500_000);
    for v in subset.iter_mut() {
        *v += 1_000_000 // it was 0-499,999. Now it is 1,000,000 - 1,499,999
    }
    // Now the queue will be 500,000 - 1,499,999
    queue.iter_mut().for_each(|v| {
        *v -= 500_000
    }); // it become two halves of 0-999,999

    queue.subset_to(700_000).into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i, *v);
    });
}

#[test]
fn test_subqueue_mut_tail_head_from_subset() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999. Now head is point at 500,000
    for i in 0..500_000 {
        assert_eq!(queue.dequeue().unwrap(), i);
    }
    // re-fill the queue again so it's 500,000 - 999,999 then 0-499,999
    for i in 0..500_000 {
        queue.enqueue(i);
    }
    // Get last 500,000 subset from queue
    let subset = queue.subset_mut(500_000, 1_000_000);
    for v in subset {
        *v += 1_000_000 // it was 0-499,999. Now it is 1,000,000 - 1,499,999
    }
    // Now the queue will be 500,000 - 1,499,999
    queue.iter_mut().for_each(|v| {
        *v -= 500_000
    }); // it become two halves of 0-999,999

    queue.subset(300_000, 800_000).into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i + 300_000, *v);
    });
}

#[test]
fn test_remove_one() {
    let mut queue = QueueAlike::with_capacity(100);
    // fill entire queue from 0-999,999
    for i in 0..100 {
        queue.enqueue(i);
    }
    queue.remove(100).unwrap(); // remove last element in this collection
    assert_eq!(queue.len(), 99);
    assert_eq!(queue.tail, 99);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i, *v);});
    queue.remove(0).unwrap(); // remove first element
    assert_eq!(queue.head, 1);
    assert_eq!(queue.len(), 98);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 1, *v);});
    queue.remove(49).unwrap(); // remove element in the middle
    assert_eq!(queue.len(), 97);
    queue.iter().enumerate().for_each(|(i, v)| {
        if i < 49 {
            assert_eq!(i + 1, *v);
        } else {
            assert_eq!(i + 2, *v);
        }
    });
    // remove half of the queue
    for i in 0..49 {
        assert_eq!(queue.dequeue().unwrap(), i + 1);
    }
    // Now queue has 51-98
    queue.iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i + 51, *v);
    });
    for i in 0..52 {
        queue.enqueue(i + 99); // Now queue will be filled by 51-150
    }

    for i in 51..=150 {
        assert_eq!(queue.dequeue().unwrap(), i);
    }
}

#[test]
fn test_dequeue_to() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // remove 0-499,999 from head of queue. Now head is point at 500,000
    queue.dequeue_to(500_000).unwrap().into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i, v);
    });
    // re-fill the queue again so it's 500,000 - 1,499,999
    for i in 1_000_000..1_500_000 {
        queue.enqueue(i);
    }

    // Remove first 600,000 element which will took last 500,000 elements and first
    // 100,000 elements from queue
    queue.dequeue_to(600_000).unwrap().into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(i + 500_000, v);
    });
}

#[test]
fn test_basic_remove_within() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    assert_eq!(queue.remove_within(..10).unwrap(), (0..10).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_990);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 10, *v)});
    assert_eq!(queue.remove_within(..=10).unwrap(), (10..=20).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_979);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 21, *v)});
    // Now, 21 elements were removed.
    // Try remove one last element
    assert_eq!(queue.remove_within(999_978..).unwrap(), vec![999_999]);
    assert_eq!(queue.len(), 999_978);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 21, *v)});
    // Try remove last 10 element
    assert_eq!(queue.remove_within(999_967..).unwrap(), (999_988..999_999).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_967);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 21, *v)});

    // Remove 21 element from center
    assert_eq!(queue.remove_within(499_979..500_000).unwrap(), (500_000..500_021).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_946);
    for i in 0..499_979 {
        assert_eq!(queue[i], i + 21);
    }
    for i in 499_979..999_946 {
        assert_eq!(queue[i], i + 42);
    }
}

#[test]
fn test_tail_head_remove_within() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    queue.dequeue_to(500_000); // Remove first 500K items so head is now point at the middle
    for i in 1_000_000..1_500_000 {
        queue.enqueue(i); // Now queue has value 500,000 - 1,499,999
    }
    queue.iter_mut().for_each(|v| {*v -= 500_000}); // Now queue has value 0 - 999,999
    assert_eq!(queue.remove_within(..10).unwrap(), (0..10).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_990);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 10, *v)});
    assert_eq!(queue.remove_within(..=10).unwrap(), (10..=20).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_979);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 21, *v)});
    // Now, 21 elements were removed.
    // Try remove one last element
    assert_eq!(queue.remove_within(999_978..).unwrap(), vec![999_999]);
    assert_eq!(queue.len(), 999_978);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 21, *v)});
    // Try remove last 10 element
    assert_eq!(queue.remove_within(999_967..).unwrap(), (999_988..999_999).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_967);
    queue.iter().enumerate().for_each(|(i, v)| {assert_eq!(i + 21, *v)});

    // Remove 22 element from center. This removal range will cross the buffer tail 
    // so it'll wrap pointer to the beginning of buffer. 
    assert_eq!(queue.remove_within(499_978..500_000).unwrap(), (499_999..500_021).map(|v| v).collect::<Vec<usize>>());
    assert_eq!(queue.len(), 999_945);
    for i in 0..499_978 {
        assert_eq!(queue[i], i + 21);
    }
    for i in 499_978..999_945 {
        assert_eq!(queue[i], i + 43);
    }
}

#[test]
fn test_basic_remove_one() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    // Remove one from front
    queue.remove_one(&0).expect("Fail to remove expected first element");
    assert_eq!(queue.len(), 999_999);
    queue.remove_one(&999_999).expect("Fail to remove expected last element");
    assert_eq!(queue.len(), 999_998);
    queue.remove_one(&499_999).expect("Fail to remove expected center element");
    assert_eq!(queue.len(), 999_997);
    for i in 0..499_998 {
        assert_eq!(queue[i], i + 1);
    }
    for i in 499_998..999_997 {
        assert_eq!(queue[i], i + 2);
    }
}

#[test]
fn test_tail_head_remove_one() {
    let mut queue = QueueAlike::with_capacity(1_000_000);
    // fill entire queue from 0-999,999
    for i in 0..1_000_000 {
        queue.enqueue(i);
    }
    queue.dequeue_to(500_000); // Remove half of it
    for i in 1_000_000..1_500_000 {
        queue.enqueue(i);
    }
    // Now, queue is 500,000-1,499,999
    queue.iter_mut().for_each(|v| {*v -= 500_000});
    // Now, queue is 0-999,999 but head is at middle of buffer
    // Remove one from front of queue but it is at the middle of buffer
    queue.remove_one(&0).expect("Fail to remove expected first element");
    assert_eq!(queue.len(), 999_999);
    // Remove one from last of queue but it is located in front of previous front in the buffer
    queue.remove_one(&999_999).expect("Fail to remove expected last element");
    assert_eq!(queue.len(), 999_998);
    // Remove one from middle of queue but it is actually first element in the buffer
    queue.remove_one(&500_000).expect("Fail to remove expected center element");
    assert_eq!(queue.len(), 999_997);
    // Remove one from middle of queue but it is actually last element in the buffer
    queue.remove_one(&499_999).expect("Fail to remove expected center element");
    assert_eq!(queue.len(), 999_996);
    for i in 0..499_998 {
        assert_eq!(queue[i], i + 1);
    }
    for i in 499_998..999_996 {
        assert_eq!(queue[i], i + 3);
    }
}

/// The test will be extremely slow. This is intended to let tester observe how
/// the queue manage memory. The memory consumption shall be constant. It shall
/// never consume more memory regardless of how long it run.
#[ignore]
#[test]
fn test_single_mem_leak() {
    fn enqueue<T>(queue: &mut QueueAlike<T>, val: T) {
        queue.enqueue(val);
    }
    fn dequeue<T>(queue: &mut QueueAlike<T>) -> T {
        queue.dequeue().unwrap()
    }
    let mut queue = QueueAlike::with_capacity(1);

    for i in 0..1_000_000_000 {
        enqueue(&mut queue, i);
        assert_eq!(dequeue(&mut queue), i);
    }
}
/// The test will be extremely slow. This is intended to let tester observe how
/// the queue manage memory. The memory consumption shall be constant. It shall
/// never consume more memory regardless of how long it run.
#[ignore]
#[test]
fn test_object_mem_leak() {
    struct ComplexType<T> {
        value: Box<T>,
        another: Vec<T>
    }
    let mut queue = QueueAlike::with_capacity(1);

    for i in 0..1_000_000_000 {
        queue.enqueue(ComplexType {
            value: Box::from(i),
            another: (i..(i + 5_000)).collect()
        });
        let obj = queue.dequeue().unwrap();
        assert_eq!(*obj.value, i);
        assert_eq!(obj.another, (i..(i + 5_000)).collect::<Vec<usize>>());
    }
}
/// The test will be extremely slow. This is intended to let tester observe how
/// the queue manage memory. The memory consumption shall be constant. It shall
/// never consume more memory regardless of how long it run.
#[ignore]
#[test]
fn test_multi_queue_mem_leak() {
    fn enqueue<T>(queue: &mut QueueAlike<T>, val: T) {
        queue.enqueue(val);
    }
    fn dequeue<T>(queue: &mut QueueAlike<T>) -> T {
        queue.dequeue().unwrap()
    }

    for i in 0..1_000_000_000 {
        let mut queue = QueueAlike::with_capacity(1);
        enqueue(&mut queue, i);
        assert_eq!(dequeue(&mut queue), i);
    }
}