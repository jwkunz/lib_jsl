//! Quicksort.

/// Sorts a slice using quicksort (median-of-three pivot, Hoare-style partition, insertion sort
/// for small ranges, recursing on the smaller side so stack depth is O(log n)).
pub fn quicksort<T: Ord>(slice: &mut [T]) {
    let mut slice = slice;
    loop {
        let n = slice.len();
        if n <= 16 {
            insertion_sort(slice);
            return;
        }
        let p = partition(slice);
        let (left, right) = slice.split_at_mut(p);
        let right = &mut right[1..];
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

/// Partitions around a median-of-three pivot; returns the pivot's final index.
fn partition<T: Ord>(s: &mut [T]) -> usize {
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
    // Median now at mid; move to the end as pivot.
    s.swap(mid, last);
    let mut store = 0;
    for i in 0..last {
        if s[i] < s[last] {
            s.swap(i, store);
            store += 1;
        }
    }
    s.swap(store, last);
    store
}

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
