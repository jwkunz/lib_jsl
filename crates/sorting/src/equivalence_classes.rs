//! Equivalence classes.
//!
//! See the book entry [Selection and equivalence classes](https://jwkunz.github.io/lib_jsl/book/sorting/selection.html#equivalence-classes).

/// Computes equivalence classes given an equivalence relation.
///
/// Uses a union-find structure over all pairs, so `equiv` is only required to hold between pairs
/// (the transitive closure is taken). Classes are ordered by their first member's position and
/// items keep their original relative order within a class.
// ANCHOR: classes
pub fn equivalence_classes<T, F>(items: &[T], equiv: F) -> Vec<Vec<T>>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    let n = items.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj && equiv(&items[i], &items[j]) {
                // Keep the smaller index as root so classes order by first member.
                if ri < rj {
                    parent[rj] = ri;
                } else {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut index_of_root: Vec<Option<usize>> = vec![None; n];
    let mut classes: Vec<Vec<T>> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let r = find(&mut parent, i);
        match index_of_root[r] {
            Some(k) => classes[k].push(item.clone()),
            None => {
                index_of_root[r] = Some(classes.len());
                classes.push(vec![item.clone()]);
            }
        }
    }
    classes
}
// ANCHOR_END: classes

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_three() {
        let v: Vec<i32> = (0..10).collect();
        let c = equivalence_classes(&v, |a, b| a % 3 == b % 3);
        assert_eq!(c, vec![vec![0, 3, 6, 9], vec![1, 4, 7], vec![2, 5, 8]]);
    }

    #[test]
    fn transitive_closure() {
        // Only adjacent values are directly related; closure joins everything.
        let v: Vec<i32> = vec![1, 2, 3, 10, 11];
        let c = equivalence_classes(&v, |a, b| (a - b).abs() == 1);
        assert_eq!(c, vec![vec![1, 2, 3], vec![10, 11]]);
    }

    #[test]
    fn empty() {
        let c = equivalence_classes::<i32, _>(&[], |a, b| a == b);
        assert!(c.is_empty());
    }
}
