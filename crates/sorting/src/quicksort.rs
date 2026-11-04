//! Quicksort.
//!
//! See the book entry [Comparison sorting](https://jwkunz.github.io/lib_jsl/book/sorting/comparison-sorts.html#quicksort).

use crate::heapsort::heapsort;

/// Sorts a slice using quicksort: a median-of-three pivot, Hoare's partition (whose two scans
/// both stop at keys equal to the pivot, so runs of equal keys are split evenly rather than piling
/// up on one side), insertion sort for ranges of 16 or fewer, and recursion on the smaller side so
/// the stack depth is `O(log n)`. As in Musser's introsort, a range still unsorted after
/// `2 log₂ n` partitions is finished by heapsort, so the worst case is `O(n log n)` even on
/// inputs built to defeat median-of-three.
pub fn quicksort<T: Ord>(slice: &mut [T]) {
    let depth = 2 * (usize::BITS - slice.len().leading_zeros()) as usize;
    introsort(slice, depth);
}

// ANCHOR: introsort
fn introsort<T: Ord>(slice: &mut [T], mut depth: usize) {
    let mut slice = slice;
    loop {
        if slice.len() <= 16 {
            insertion_sort(slice);
            return;
        }
        if depth == 0 {
            // Too many unbalanced partitions: the pivots are being chosen badly.
            heapsort(slice);
            return;
        }
        depth -= 1;
        let p = partition(slice);
        let (left, rest) = slice.split_at_mut(p);
        let right = &mut rest[1..];
        if left.len() < right.len() {
            introsort(left, depth);
            slice = right;
        } else {
            introsort(right, depth);
            slice = left;
        }
    }
}
// ANCHOR_END: introsort

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

    /// McIlroy's adversary: keys start as "gas", larger than every value, and are frozen to
    /// increasing values only when the sort compares two gas keys, choosing which one so as to
    /// make the current pivot as bad as possible.
    mod adversary {
        use std::cell::RefCell;
        use std::cmp::Ordering;
        use std::rc::Rc;

        pub struct State {
            pub val: Vec<usize>,
            gas: usize,
            solid: usize,
            candidate: usize,
            pub comparisons: u64,
        }

        #[derive(Clone)]
        pub struct Key {
            pub i: usize,
            pub state: Rc<RefCell<State>>,
        }

        pub fn keys(n: usize) -> (Vec<Key>, Rc<RefCell<State>>) {
            let state = Rc::new(RefCell::new(State { val: vec![n; n], gas: n, solid: 0, candidate: 0, comparisons: 0 }));
            ((0..n).map(|i| Key { i, state: state.clone() }).collect(), state)
        }

        impl PartialEq for Key {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == Ordering::Equal
            }
        }
        impl Eq for Key {}
        impl PartialOrd for Key {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Key {
            fn cmp(&self, other: &Self) -> Ordering {
                let mut s = self.state.borrow_mut();
                s.comparisons += 1;
                let (x, y, gas) = (self.i, other.i, s.gas);
                if s.val[x] == gas && s.val[y] == gas {
                    let frozen = if x == s.candidate { x } else { y };
                    s.val[frozen] = s.solid;
                    s.solid += 1;
                }
                if s.val[x] == gas {
                    s.candidate = x;
                } else if s.val[y] == gas {
                    s.candidate = y;
                }
                s.val[x].cmp(&s.val[y])
            }
        }
    }

    #[test]
    fn an_adversary_cannot_force_quadratic_time() {
        // Without the heapsort fallback the adversary forced about n²/4 comparisons
        // (16 million for n = 8000, 154 n log₂ n).
        let n = 8000;
        let (mut keys, state) = adversary::keys(n);
        quicksort(&mut keys);
        let comparisons = state.borrow().comparisons as f64;
        assert!(comparisons < 8.0 * n as f64 * (n as f64).log2(), "{comparisons} comparisons");
        let val = state.borrow().val.clone();
        assert!(keys.windows(2).all(|w| val[w[0].i] <= val[w[1].i]));
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
