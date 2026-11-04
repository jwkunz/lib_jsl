//! Heapsort.
//!
//! See the book entry [Comparison sorting](https://jwkunz.github.io/lib_jsl/book/sorting/comparison-sorts.html#heapsort).

/// Sorts a slice using heapsort (in place, O(n log n) worst case).
// ANCHOR: heapsort
pub fn heapsort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    for start in (0..n / 2).rev() {
        sift_down(slice, start, n);
    }
    for end in (1..n).rev() {
        slice.swap(0, end);
        sift_down(slice, 0, end);
    }
}

fn sift_down<T: Ord>(s: &mut [T], mut root: usize, end: usize) {
    loop {
        let mut child = 2 * root + 1;
        if child >= end {
            return;
        }
        if child + 1 < end && s[child] < s[child + 1] {
            child += 1;
        }
        if s[root] >= s[child] {
            return;
        }
        s.swap(root, child);
        root = child;
    }
}
// ANCHOR_END: heapsort

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts() {
        let mut v: Vec<i32> = (0..1000).map(|i| (i * 7919) % 1009 - 500).collect();
        let mut expected = v.clone();
        expected.sort();
        heapsort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn edge_cases() {
        let mut e: Vec<i32> = vec![];
        heapsort(&mut e);
        let mut two = vec![2, 1];
        heapsort(&mut two);
        assert_eq!(two, vec![1, 2]);
    }
}
