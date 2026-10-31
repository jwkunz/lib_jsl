//! Select M largest.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Selects the `m` largest elements from a slice, returned in descending order.
///
/// Uses a bounded min-heap, so it runs in O(n log m) time and O(m) extra space. If `m >= len`
/// all elements are returned (sorted descending).
pub fn select_m_largest<T: Ord + Clone>(slice: &[T], m: usize) -> Vec<T> {
    if m == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Reverse<T>> = BinaryHeap::with_capacity(m + 1);
    for item in slice {
        if heap.len() < m {
            heap.push(Reverse(item.clone()));
        } else if let Some(Reverse(min)) = heap.peek() {
            if item > min {
                heap.pop();
                heap.push(Reverse(item.clone()));
            }
        }
    }
    let mut out: Vec<T> = heap.into_iter().map(|Reverse(x)| x).collect();
    out.sort_by(|a, b| b.cmp(a));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_largest() {
        let v = vec![5, 1, 9, 3, 9, 7, 2];
        assert_eq!(select_m_largest(&v, 3), vec![9, 9, 7]);
    }

    #[test]
    fn edge_cases() {
        let v = vec![2, 1, 3];
        assert!(select_m_largest(&v, 0).is_empty());
        assert_eq!(select_m_largest(&v, 10), vec![3, 2, 1]);
        assert!(select_m_largest::<i32>(&[], 3).is_empty());
    }
}
