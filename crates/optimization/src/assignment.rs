//! The linear sum assignment problem (SciPy's `linear_sum_assignment`): match rows to columns of
//! a cost matrix, one-to-one, to minimise (or maximise) the total cost, by the shortest
//! augmenting path algorithm of Jonker and Volgenant as described by Crouse (2016).
//!
//! See the book entry [Assignment, trust regions and global optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/assignment-trust-global.html#the-assignment-problem).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// One shortest augmenting path from row `i` in the reduced costs `c[i][j] − u[i] − v[j]`
/// (Dijkstra's method over the columns): returns the free column it ends at and the path length.
#[allow(clippy::too_many_arguments)]
fn augmenting_path(nc: usize, cost: &[f64], u: &[f64], v: &[f64], path: &mut [usize], row4col: &[Option<usize>], shortest: &mut [f64], mut i: usize, sr: &mut [bool], sc: &mut [bool], remaining: &mut [usize]) -> Option<(usize, f64)> {
    let mut min_val = 0.0;
    let mut num_remaining = nc;
    // Reverse order, so that a constant cost matrix gives the identity (SciPy issue 11602).
    for (it, r) in remaining.iter_mut().enumerate() {
        *r = nc - it - 1;
    }
    sr.fill(false);
    sc.fill(false);
    shortest.fill(f64::INFINITY);
    loop {
        let mut index = usize::MAX;
        let mut lowest = f64::INFINITY;
        sr[i] = true;
        for (it, &j) in remaining.iter().enumerate().take(num_remaining) {
            let r = min_val + cost[i * nc + j] - u[i] - v[j];
            if r < shortest[j] {
                path[j] = i;
                shortest[j] = r;
            }
            // Among equal minima prefer a free column, which ends the search.
            if shortest[j] < lowest || (shortest[j] == lowest && row4col[j].is_none()) {
                lowest = shortest[j];
                index = it;
            }
        }
        min_val = lowest;
        if min_val == f64::INFINITY {
            return None;
        }
        let j = remaining[index];
        sc[j] = true;
        num_remaining -= 1;
        remaining[index] = remaining[num_remaining];
        match row4col[j] {
            None => return Some((j, min_val)),
            Some(r) => i = r,
        }
    }
}

// ANCHOR: lsap
/// The optimal assignment: pairs `(rowᵢ, colᵢ)`, as many as the smaller dimension, sorted by
/// row, minimising `Σ cost[rowᵢ][colᵢ]` (maximising with `maximize`). Each row in turn is joined
/// to the current matching by a shortest augmenting path in the reduced costs, and the dual
/// variables `u`, `v` are updated so that all reduced costs stay non-negative (the Hungarian
/// method's invariant). A tall matrix is transposed first. `O(n³)`. Infinite costs forbid a
/// pair; an error is returned if no complete assignment avoids them, or for NaN or `−∞`.
pub fn linear_sum_assignment(cost: &Array2<f64>, maximize: bool) -> Result<(Vec<usize>, Vec<usize>), ErrorsJSL> {
    let (mut nr, mut nc) = cost.dim();
    if nr == 0 || nc == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let transpose = nc < nr;
    let mut c: Vec<f64> = if transpose { cost.t().iter().copied().collect() } else { cost.iter().copied().collect() };
    if transpose {
        std::mem::swap(&mut nr, &mut nc);
    }
    if maximize {
        c.iter_mut().for_each(|x| *x = -*x);
    }
    if c.iter().any(|x| x.is_nan() || *x == f64::NEG_INFINITY) {
        return Err(ErrorsJSL::InvalidInputRange("the cost matrix contains NaN or −∞"));
    }
    let (mut u, mut v) = (vec![0.0; nr], vec![0.0; nc]);
    let mut shortest = vec![0.0; nc];
    let mut path = vec![0usize; nc];
    let mut col4row: Vec<Option<usize>> = vec![None; nr];
    let mut row4col: Vec<Option<usize>> = vec![None; nc];
    let (mut sr, mut sc) = (vec![false; nr], vec![false; nc]);
    let mut remaining = vec![0usize; nc];
    for cur in 0..nr {
        let (sink, min_val) = augmenting_path(nc, &c, &u, &v, &mut path, &row4col, &mut shortest, cur, &mut sr, &mut sc, &mut remaining).ok_or(ErrorsJSL::InvalidInputRange("the cost matrix is infeasible"))?;
        u[cur] += min_val;
        for i in 0..nr {
            if sr[i] && i != cur {
                u[i] += min_val - shortest[col4row[i].expect("matched")];
            }
        }
        for j in 0..nc {
            if sc[j] {
                v[j] -= min_val - shortest[j];
            }
        }
        // Augment along the path back to the current row.
        let mut j = sink;
        loop {
            let i = path[j];
            row4col[j] = Some(i);
            let previous = col4row[i].replace(j);
            if i == cur {
                break;
            }
            j = previous.expect("rows on the path are matched");
        }
    }
    let pairs: Vec<(usize, usize)> = if transpose {
        let mut p: Vec<(usize, usize)> = col4row.iter().enumerate().map(|(r, cl)| (cl.expect("matched"), r)).collect();
        p.sort_unstable();
        p
    } else {
        col4row.iter().enumerate().map(|(r, cl)| (r, cl.expect("matched"))).collect()
    };
    Ok(pairs.into_iter().unzip())
}
// ANCHOR_END: lsap

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_examples() {
        // SciPy's documentation example.
        let cost = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0]).unwrap();
        let (r, c) = linear_sum_assignment(&cost, false).unwrap();
        assert_eq!(r, vec![0, 1, 2]);
        assert_eq!(c, vec![1, 0, 2]);
        let total: f64 = r.iter().zip(&c).map(|(&i, &j)| cost[(i, j)]).sum();
        assert_eq!(total, 5.0);
        let (_, c) = linear_sum_assignment(&Array2::from_elem((4, 4), 1.0), false).unwrap();
        assert_eq!(c, vec![0, 1, 2, 3]);
    }

    #[test]
    fn brute_force_agrees_on_rectangles() {
        let mut s: u64 = 7;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 100) as f64
        };
        for (nr, nc) in [(4, 4), (3, 5), (5, 3)] {
            let cost = Array2::from_shape_fn((nr, nc), |_| next());
            for maximize in [false, true] {
                let (r, c) = linear_sum_assignment(&cost, maximize).unwrap();
                let got: f64 = r.iter().zip(&c).map(|(&i, &j)| cost[(i, j)]).sum();
                // Brute force over injective maps from the smaller side.
                let (small, big) = (nr.min(nc), nr.max(nc));
                let mut best = if maximize { f64::NEG_INFINITY } else { f64::INFINITY };
                let mut perm: Vec<usize> = (0..big).collect();
                fn permute(k: usize, small: usize, perm: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
                    if k == small {
                        f(&perm[..small]);
                        return;
                    }
                    for i in k..perm.len() {
                        perm.swap(k, i);
                        permute(k + 1, small, perm, f);
                        perm.swap(k, i);
                    }
                }
                permute(0, small, &mut perm, &mut |p| {
                    let total: f64 = (0..small).map(|a| if nr <= nc { cost[(a, p[a])] } else { cost[(p[a], a)] }).sum();
                    best = if maximize { best.max(total) } else { best.min(total) };
                });
                assert_eq!(got, best, "{nr} × {nc}, maximize {maximize}");
            }
        }
    }
}
