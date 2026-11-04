//! Quicksort.

/// Sorts a slice using quicksort: a median-of-three pivot, Hoare's partition (whose two scans
/// both stop at keys equal to the pivot, so runs of equal keys are split evenly rather than piling
/// up on one side), insertion sort for ranges of 16 or fewer, and recursion on the smaller side so
/// the stack depth is `O(log n)`. `O(n log n)` on average and on sorted, reversed or
/// repeated-key input; inputs built against median-of-three can still take `O(n²)`.
pub fn quicksort<T: Ord>(slice: &mut [T]) {
    let mut slice = slice;
    loop {
        let n = slice.len();
        if n <= 16 {
            insertion_sort(slice);
            return;
        }
        let p = partition(slice);
        let (left, rest) = slice.split_at_mut(p);
        let right = &mut rest[1..];
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
/// Hoare's partition around a median-of-three pivot; returns the pivot's final index, with
/// smaller-or-equal keys before it and greater-or-equal keys after.
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
    // Now s[0] <= s[mid] <= s[last]. The median becomes the pivot at index 0, where it stops the
    // right-to-left scan, and s[last] >= pivot stops the left-to-right one.
    s.swap(0, mid);
    let (mut i, mut j) = (0, s.len());
    loop {
        i += 1;
        while s[i] < s[0] {
            i += 1;
        }
        j -= 1;
        while s[0] < s[j] {
            j -= 1;
        }
        if i >= j {
            break;
        }
        s.swap(i, j);
    }
    s.swap(0, j);
    j
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
    fn equal_keys_and_presorted_input_stay_fast() {
        // A partition that sends every key equal to the pivot to one side takes quadratic time on
        // these, and one that rearranges the greater keys can do so on sorted input.
        let mut same = vec![7u8; 200_000];
        quicksort(&mut same);
        assert!(same.iter().all(|&x| x == 7));
        let mut few: Vec<u32> = (0..200_000u32).map(|i| i.wrapping_mul(2_654_435_761) % 3).collect();
        let mut expected = few.clone();
        expected.sort();
        quicksort(&mut few);
        assert_eq!(few, expected);
        let mut organ: Vec<u32> = (0..200_000u32).map(|i| i.min(200_000 - i)).collect();
        let mut expected = organ.clone();
        expected.sort();
        quicksort(&mut organ);
        assert_eq!(organ, expected);
        let mut sorted: Vec<u32> = (0..200_000).collect();
        quicksort(&mut sorted);
        assert!(sorted.windows(2).all(|w| w[0] < w[1]));
        let mut reversed: Vec<u32> = (0..200_000).rev().collect();
        quicksort(&mut reversed);
        assert_eq!(reversed, sorted);
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
