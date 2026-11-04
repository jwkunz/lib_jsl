//! Quicksort.

/// Sorts a slice using quicksort: a median-of-three pivot, a three-way partition into keys less
/// than, equal to and greater than the pivot (so runs of equal keys cost linear time), insertion
/// sort for ranges of 16 or fewer, and recursion on the smaller side so the stack depth is
/// `O(log n)`. `O(n log n)` on average; inputs built against median-of-three can still take
/// `O(n²)`.
pub fn quicksort<T: Ord>(slice: &mut [T]) {
    let mut slice = slice;
    loop {
        let n = slice.len();
        if n <= 16 {
            insertion_sort(slice);
            return;
        }
        let (lt, gt) = partition(slice);
        let (left, rest) = slice.split_at_mut(lt);
        let right = &mut rest[gt - lt..];
        if left.len() < right.len() {
            quicksort(left);
            slice = right;
        } else {
            quicksort(right);
            slice = left;
        }
    }
}

fn insertion_sort<T: Ord>(s: &mut [T]) {
    for i in 1..s.len() {
        let mut j = i;
        while j > 0 && s[j - 1] > s[j] {
            s.swap(j - 1, j);
            j -= 1;
        }
    }
}

// ANCHOR: partition
/// Partitions around a median-of-three pivot into `s[..lt] < pivot`, `s[lt..gt] == pivot` and
/// `s[gt..] > pivot` (Dijkstra's "Dutch national flag"); returns `(lt, gt)`.
fn partition<T: Ord>(s: &mut [T]) -> (usize, usize) {
    let last = s.len() - 1;
    let mid = last / 2;
    if s[mid] < s[0] {
        s.swap(mid, 0);
    }
    if s[last] < s[0] {
        s.swap(last, 0);
    }
    if s[last] < s[mid] {
        s.swap(last, mid);
    }
    // The median is now at `mid`; keep the pivot at index 0 while scanning.
    s.swap(0, mid);
    let (mut lt, mut i, mut gt) = (0, 1, s.len());
    // Invariant: s[..lt] < pivot = s[lt], s[lt..i] == pivot, s[gt..] > pivot.
    while i < gt {
        match s[i].cmp(&s[lt]) {
            std::cmp::Ordering::Less => {
                s.swap(i, lt);
                lt += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                gt -= 1;
                s.swap(i, gt);
            }
            std::cmp::Ordering::Equal => i += 1,
        }
    }
    (lt, gt)
}
// ANCHOR_END: partition

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_various() {
        let mut v: Vec<i32> = (0..1000).map(|i| (i * 7919) % 1009 - 500).collect();
        let mut expected = v.clone();
        expected.sort();
        quicksort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn many_equal_keys_take_linear_time_per_level() {
        // With a two-way partition all keys equal to the pivot went to one side, so these took
        // quadratic time; the three-way partition finishes each in one pass.
        let mut same = vec![7u8; 200_000];
        quicksort(&mut same);
        assert!(same.iter().all(|&x| x == 7));
        let mut few: Vec<u32> = (0..200_000u32).map(|i| i.wrapping_mul(2_654_435_761) % 3).collect();
        let mut expected = few.clone();
        expected.sort();
        quicksort(&mut few);
        assert_eq!(few, expected);
    }

    #[test]
    fn edge_cases() {
        let mut e: Vec<i32> = vec![];
        quicksort(&mut e);
        let mut one = vec![1];
        quicksort(&mut one);
        let mut dup = vec![5; 100];
        quicksort(&mut dup);
        assert_eq!(dup, vec![5; 100]);
        let mut asc: Vec<i32> = (0..500).collect();
        quicksort(&mut asc);
        assert!(asc.windows(2).all(|w| w[0] <= w[1]));
        let mut desc: Vec<i32> = (0..500).rev().collect();
        quicksort(&mut desc);
        assert!(desc.windows(2).all(|w| w[0] <= w[1]));
    }
}
