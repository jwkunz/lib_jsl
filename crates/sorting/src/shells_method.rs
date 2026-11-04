//! Shell's method.
//!
//! See the book entry [Comparison sorting](https://jwkunz.github.io/lib_jsl/book/sorting/comparison-sorts.html#shells-method).

/// Sorts a slice using Shell's method with Ciura's gap sequence extended by a factor of 2.25.
// ANCHOR: shell
pub fn shell_sort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    let mut gaps: Vec<usize> = vec![1, 4, 10, 23, 57, 132, 301, 701];
    while *gaps.last().unwrap() < n {
        let next = (*gaps.last().unwrap() as f64 * 2.25) as usize;
        gaps.push(next);
    }
    for &gap in gaps.iter().rev() {
        if gap >= n {
            continue;
        }
        for i in gap..n {
            let mut j = i;
            while j >= gap && slice[j - gap] > slice[j] {
                slice.swap(j - gap, j);
                j -= gap;
            }
        }
    }
}
// ANCHOR_END: shell

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts() {
        let mut v: Vec<i32> = (0..3000).map(|i| (i * 7919) % 1009 - 500).collect();
        let mut expected = v.clone();
        expected.sort();
        shell_sort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn edge_cases() {
        let mut e: Vec<i32> = vec![];
        shell_sort(&mut e);
        let mut one = vec![3];
        shell_sort(&mut one);
        assert_eq!(one, vec![3]);
    }
}
