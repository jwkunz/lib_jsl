//! Dynamic programming: classic optimisation problems solved by tabulation, and solvers for
//! Markov decision processes.
//!
//! The first group (knapsack, coin change, longest common / increasing subsequence, edit distance,
//! matrix-chain ordering) each return the optimal value together with a reconstruction of one
//! optimal solution. The second group, [`FiniteMdp`], solves sequential decision problems by
//! backward induction (finite horizon) or value iteration (discounted, infinite horizon).
//!
//! See the book entry [Dynamic programming](https://jwkunz.github.io/lib_jsl/book/optimization/dynamic-programming.html).

use lib_jsl_core::ErrorsJSL;

/// 0/1 knapsack: chooses items (each at most once) maximising total value within `capacity`.
/// Returns `(best value, chosen item indices in increasing order)`. `O(n · capacity)` time and
/// memory.
pub fn knapsack_01(weights: &[usize], values: &[f64], capacity: usize) -> Result<(f64, Vec<usize>), ErrorsJSL> {
    if weights.len() != values.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((weights.len(), values.len())));
    }
    let n = weights.len();
    // best[i][w] = best value using the first i items with capacity w.
    // ANCHOR: knapsack
    let mut best = vec![vec![0.0f64; capacity + 1]; n + 1];
    for i in 1..=n {
        for w in 0..=capacity {
            best[i][w] = best[i - 1][w];
            if weights[i - 1] <= w {
                best[i][w] = best[i][w].max(best[i - 1][w - weights[i - 1]] + values[i - 1]);
            }
        }
    }
    let mut chosen = Vec::new();
    let mut w = capacity;
    for i in (1..=n).rev() {
        if best[i][w] != best[i - 1][w] {
            chosen.push(i - 1);
            w -= weights[i - 1];
        }
    }
    chosen.reverse();
    Ok((best[n][capacity], chosen))
    // ANCHOR_END: knapsack
}

/// The fewest coins (unlimited supply of each denomination) summing to `amount`, as the count of
/// each denomination; `None` if the amount cannot be made. Zero-valued denominations are ignored.
pub fn coin_change_min_coins(coins: &[usize], amount: usize) -> Option<Vec<usize>> {
    let mut best: Vec<Option<(usize, usize)>> = vec![None; amount + 1]; // (coins used, last coin index)
    best[0] = Some((0, 0));
    for a in 1..=amount {
        for (ci, &c) in coins.iter().enumerate() {
            if c > 0
                && c <= a
                && let Some((used, _)) = best[a - c]
                && best[a].is_none_or(|(u, _)| used + 1 < u)
            {
                best[a] = Some((used + 1, ci));
            }
        }
    }
    best[amount]?;
    let mut counts = vec![0; coins.len()];
    let mut a = amount;
    while a > 0 {
        let (_, ci) = best[a]?;
        counts[ci] += 1;
        a -= coins[ci];
    }
    Some(counts)
}

/// The number of distinct multisets of coins summing to `amount` (order ignored), saturating at
/// `u64::MAX`.
pub fn coin_change_ways(coins: &[usize], amount: usize) -> u64 {
    let mut ways = vec![0u64; amount + 1];
    ways[0] = 1;
    for &c in coins.iter().filter(|&&c| c > 0) {
        for a in c..=amount {
            ways[a] = ways[a].saturating_add(ways[a - c]);
        }
    }
    ways[amount]
}

/// A longest common subsequence of `a` and `b` as pairs of matched indices `(i in a, j in b)`.
pub fn longest_common_subsequence<T: PartialEq>(a: &[T], b: &[T]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut len = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            len[i][j] = if a[i - 1] == b[j - 1] { len[i - 1][j - 1] + 1 } else { len[i - 1][j].max(len[i][j - 1]) };
        }
    }
    let (mut i, mut j) = (n, m);
    let mut pairs = Vec::with_capacity(len[n][m]);
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if len[i - 1][j] >= len[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// One step of an edit script turning sequence `a` into `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOp {
    /// `a[i]` equals `b[j]`: keep it.
    Keep { i: usize, j: usize },
    /// Replace `a[i]` with `b[j]`.
    Substitute { i: usize, j: usize },
    /// Insert `b[j]`.
    Insert { j: usize },
    /// Delete `a[i]`.
    Delete { i: usize },
}

/// Levenshtein distance between `a` and `b` (unit cost for insert, delete and substitute) and one
/// optimal edit script.
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> (usize, Vec<EditOp>) {
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    let (mut i, mut j) = (n, m);
    let mut script = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]) {
            script.push(if a[i - 1] == b[j - 1] { EditOp::Keep { i: i - 1, j: j - 1 } } else { EditOp::Substitute { i: i - 1, j: j - 1 } });
            i -= 1;
            j -= 1;
        } else if i > 0 && d[i][j] == d[i - 1][j] + 1 {
            script.push(EditOp::Delete { i: i - 1 });
            i -= 1;
        } else {
            script.push(EditOp::Insert { j: j - 1 });
            j -= 1;
        }
    }
    script.reverse();
    (d[n][m], script)
}

/// The cheapest way to multiply a chain of matrices where matrix `k` is `dims[k] × dims[k+1]`:
/// `(scalar multiplications, parenthesisation such as "((A0A1)A2)")`. Needs at least one matrix
/// (`dims.len() >= 2`).
pub fn matrix_chain_order(dims: &[usize]) -> Result<(u64, String), ErrorsJSL> {
    if dims.len() < 2 {
        return Err(ErrorsJSL::InvalidInputRange("At least one matrix (two dimensions) is required."));
    }
    let n = dims.len() - 1;
    let mut cost = vec![vec![0u64; n]; n];
    let mut split = vec![vec![0usize; n]; n];
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            cost[i][j] = u64::MAX;
            for k in i..j {
                let c = cost[i][k] + cost[k + 1][j] + (dims[i] * dims[k + 1] * dims[j + 1]) as u64;
                if c < cost[i][j] {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }
    fn paren(split: &[Vec<usize>], i: usize, j: usize) -> String {
        if i == j {
            format!("A{i}")
        } else {
            format!("({}{})", paren(split, i, split[i][j]), paren(split, split[i][j] + 1, j))
        }
    }
    Ok((cost[0][n - 1], paren(&split, 0, n - 1)))
}

/// A longest strictly increasing subsequence, as indices into `values`, in `O(n log n)`.
pub fn longest_increasing_subsequence<T: PartialOrd>(values: &[T]) -> Vec<usize> {
    let mut tails: Vec<usize> = Vec::new(); // tails[k] = index of the smallest tail of an increasing run of length k+1
    let mut prev = vec![usize::MAX; values.len()];
    for (i, v) in values.iter().enumerate() {
        let pos = tails.partition_point(|&t| values[t] < *v);
        if pos > 0 {
            prev[i] = tails[pos - 1];
        }
        if pos == tails.len() {
            tails.push(i);
        } else {
            tails[pos] = i;
        }
    }
    let mut out = Vec::with_capacity(tails.len());
    let mut cur = tails.last().copied();
    while let Some(i) = cur {
        out.push(i);
        cur = (prev[i] != usize::MAX).then_some(prev[i]);
    }
    out.reverse();
    out
}

/// A finite Markov decision process: in state `s`, taking action `a` yields `rewards[s][a]` and
/// moves to state `s'` with the probabilities in `transitions[s][a]` (a list of `(s', prob)`).
/// Every state must offer at least one action; the probabilities of each action must sum to 1.
#[derive(Debug, Clone)]
pub struct FiniteMdp {
    pub transitions: Vec<Vec<Vec<(usize, f64)>>>,
    pub rewards: Vec<Vec<f64>>,
}

/// The value function and a greedy policy.
#[derive(Debug, Clone, PartialEq)]
pub struct MdpSolution {
    pub values: Vec<f64>,
    /// The maximising action in each state.
    pub policy: Vec<usize>,
}

impl FiniteMdp {
    /// Checks the structural requirements above.
    pub fn validate(&self) -> Result<(), ErrorsJSL> {
        let n = self.transitions.len();
        if n == 0 || self.rewards.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((n, self.rewards.len())));
        }
        for s in 0..n {
            if self.transitions[s].is_empty() || self.transitions[s].len() != self.rewards[s].len() {
                return Err(ErrorsJSL::Misconfiguration("Each state needs one or more actions with matching rewards."));
            }
            for outcomes in &self.transitions[s] {
                let total: f64 = outcomes.iter().map(|o| o.1).sum();
                if outcomes.iter().any(|&(t, p)| t >= n || !(0.0..=1.0).contains(&p)) || (total - 1.0).abs() > 1e-9 {
                    return Err(ErrorsJSL::Misconfiguration("Transition probabilities must be valid and sum to one."));
                }
            }
        }
        Ok(())
    }

    fn q_value(&self, s: usize, a: usize, next_values: &[f64], discount: f64) -> f64 {
        self.rewards[s][a] + discount * self.transitions[s][a].iter().map(|&(t, p)| p * next_values[t]).sum::<f64>()
    }

    fn greedy(&self, values: &[f64], discount: f64) -> MdpSolution {
        let mut policy = Vec::with_capacity(values.len());
        let mut new_values = Vec::with_capacity(values.len());
        for s in 0..values.len() {
            let (a, v) = (0..self.rewards[s].len())
                .map(|a| (a, self.q_value(s, a, values, discount)))
                .fold((0, f64::NEG_INFINITY), |best, c| if c.1 > best.1 { c } else { best });
            policy.push(a);
            new_values.push(v);
        }
        MdpSolution { values: new_values, policy }
    }

    /// Finite-horizon backward induction: the optimal expected total reward over `horizon` steps
    /// plus `terminal_values[s]` on arrival in the final state. Returns, for each stage from the
    /// first (`0`) to the last (`horizon - 1`), the value of being in each state and the optimal
    /// action there; the first element holds the optimal values over the whole horizon.
    pub fn backward_induction(&self, horizon: usize, terminal_values: &[f64]) -> Result<Vec<MdpSolution>, ErrorsJSL> {
        self.validate()?;
        if terminal_values.len() != self.rewards.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((terminal_values.len(), self.rewards.len())));
        }
        let mut stages = Vec::with_capacity(horizon);
        let mut next = terminal_values.to_vec();
        for _ in 0..horizon {
            let sol = self.greedy(&next, 1.0);
            next = sol.values.clone();
            stages.push(sol);
        }
        stages.reverse();
        Ok(stages)
    }

    /// Discounted infinite-horizon value iteration with discount in `[0, 1)`. Converges when the
    /// largest change in value falls below `tolerance`; fails if `max_iterations` is exhausted.
    pub fn value_iteration(&self, discount: f64, tolerance: f64, max_iterations: usize) -> Result<MdpSolution, ErrorsJSL> {
        self.validate()?;
        if !(0.0..1.0).contains(&discount) {
            return Err(ErrorsJSL::InvalidInputRange("Discount must be in [0, 1)."));
        }
        // ANCHOR: value_iteration
        let mut values = vec![0.0; self.rewards.len()];
        for _ in 0..max_iterations {
            let sol = self.greedy(&values, discount);
            let change = sol.values.iter().zip(&values).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            values = sol.values;
            // Standard stopping rule guaranteeing the value error is below `tolerance`.
            if change <= tolerance * (1.0 - discount) / (2.0 * discount).max(1e-12) {
                return Ok(self.greedy(&values, discount));
            }
        }
        Err(ErrorsJSL::RuntimeError("Value iteration did not converge."))
        // ANCHOR_END: value_iteration
    }

    /// The expected discounted return of following `policy` forever, by Gauss-Seidel sweeps,
    /// stopping when a sweep changes no value by `tolerance` or more (the remaining error can be
    /// up to about `discount / (1 − discount)` times larger).
    pub fn evaluate_policy(&self, policy: &[usize], discount: f64, tolerance: f64) -> Result<Vec<f64>, ErrorsJSL> {
        self.validate()?;
        if policy.len() != self.rewards.len() || policy.iter().zip(&self.rewards).any(|(&a, r)| a >= r.len()) || !(0.0..1.0).contains(&discount) {
            return Err(ErrorsJSL::InvalidInputRange("Invalid policy or discount."));
        }
        let mut v = vec![0.0; policy.len()];
        for _ in 0..1_000_000 {
            let mut change: f64 = 0.0;
            for s in 0..v.len() {
                let new = self.q_value(s, policy[s], &v, discount);
                change = change.max((new - v[s]).abs());
                v[s] = new;
            }
            if change < tolerance {
                return Ok(v);
            }
        }
        Err(ErrorsJSL::RuntimeError("Policy evaluation did not converge."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knapsack_and_coins() {
        let (best, items) = knapsack_01(&[1, 3, 4, 5], &[1.0, 4.0, 5.0, 7.0], 7).unwrap();
        assert_eq!(best, 9.0);
        assert_eq!(items, vec![1, 2]); // weights 3 + 4, values 4 + 5
        assert_eq!(knapsack_01(&[5], &[3.0], 4).unwrap(), (0.0, vec![]));
        assert!(knapsack_01(&[1], &[], 3).is_err());
        let counts = coin_change_min_coins(&[1, 5, 10, 25], 63).unwrap();
        assert_eq!(counts, vec![3, 0, 1, 2]);
        assert_eq!(coin_change_min_coins(&[4, 6], 7), None);
        assert_eq!(coin_change_min_coins(&[3, 7], 0), Some(vec![0, 0]));
        // A greedy-fooling instance: 6 = 3 + 3, not 4 + 1 + 1.
        assert_eq!(coin_change_min_coins(&[1, 3, 4], 6).unwrap().iter().sum::<usize>(), 2);
        assert_eq!(coin_change_ways(&[1, 2, 5], 5), 4);
        assert_eq!(coin_change_ways(&[2], 3), 0);
    }

    #[test]
    fn sequence_alignment_problems() {
        let a: Vec<char> = "AGGTAB".chars().collect();
        let b: Vec<char> = "GXTXAYB".chars().collect();
        let lcs = longest_common_subsequence(&a, &b);
        assert_eq!(lcs.iter().map(|&(i, _)| a[i]).collect::<String>(), "GTAB");
        assert!(lcs.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1));
        let (d, script) = edit_distance(&"kitten".chars().collect::<Vec<_>>(), &"sitting".chars().collect::<Vec<_>>());
        assert_eq!(d, 3);
        assert_eq!(script.iter().filter(|op| !matches!(op, EditOp::Keep { .. })).count(), 3);
        // Applying the script to "kitten" must produce "sitting".
        let (ka, kb): (Vec<char>, Vec<char>) = ("kitten".chars().collect(), "sitting".chars().collect());
        let rebuilt: String = script
            .iter()
            .filter_map(|op| match *op {
                EditOp::Keep { i, .. } => Some(ka[i]),
                EditOp::Substitute { j, .. } | EditOp::Insert { j } => Some(kb[j]),
                EditOp::Delete { .. } => None,
            })
            .collect();
        assert_eq!(rebuilt, "sitting");
        assert_eq!(edit_distance::<u8>(&[], &[1, 2]).0, 2);
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 2, 3]).1.len(), 3);
        let lis = longest_increasing_subsequence(&[10, 9, 2, 5, 3, 7, 101, 18]);
        assert_eq!(lis.len(), 4);
        assert!(lis.windows(2).all(|w| w[0] < w[1]));
        assert!(longest_increasing_subsequence::<i32>(&[]).is_empty());
        assert_eq!(longest_increasing_subsequence(&[3, 3, 3]).len(), 1);
    }

    #[test]
    fn matrix_chain() {
        // CLRS example: 30x35, 35x15, 15x5, 5x10, 10x20, 20x25 -> 15125 multiplications.
        let (cost, order) = matrix_chain_order(&[30, 35, 15, 5, 10, 20, 25]).unwrap();
        assert_eq!(cost, 15125);
        assert_eq!(order, "((A0(A1A2))((A3A4)A5))");
        assert_eq!(matrix_chain_order(&[4, 5]).unwrap(), (0, "A0".to_string()));
        assert!(matrix_chain_order(&[3]).is_err());
    }

    fn inventory_like() -> FiniteMdp {
        // State 0 = "poor", 1 = "rich". Action 0 = save (safe), action 1 = gamble.
        FiniteMdp {
            transitions: vec![vec![vec![(0, 1.0)], vec![(1, 0.5), (0, 0.5)]], vec![vec![(1, 1.0)], vec![(1, 0.8), (0, 0.2)]]],
            rewards: vec![vec![0.0, 0.0], vec![1.0, 2.0]],
        }
    }

    #[test]
    fn mdp_backward_induction_and_value_iteration() {
        let mdp = inventory_like();
        let stages = mdp.backward_induction(1, &[0.0, 0.0]).unwrap();
        assert_eq!(stages[0].values, vec![0.0, 2.0]); // one step: gamble when rich
        assert_eq!(stages[0].policy[1], 1);
        let long = mdp.backward_induction(30, &[0.0, 0.0]).unwrap();
        assert_eq!(long.len(), 30);
        let sol = mdp.value_iteration(0.9, 1e-10, 10_000).unwrap();
        // Bellman optimality holds for the returned values.
        for s in 0..2 {
            let best = (0..2).map(|a| mdp.rewards[s][a] + 0.9 * mdp.transitions[s][a].iter().map(|&(t, p)| p * sol.values[t]).sum::<f64>()).fold(f64::MIN, f64::max);
            assert!((best - sol.values[s]).abs() < 1e-8);
        }
        // Following the optimal policy reproduces the optimal values.
        let v = mdp.evaluate_policy(&sol.policy, 0.9, 1e-12).unwrap();
        assert!(v.iter().zip(&sol.values).all(|(a, b)| (a - b).abs() < 1e-7));
        // A worse fixed policy does no better.
        let always_save = mdp.evaluate_policy(&[0, 0], 0.9, 1e-12).unwrap();
        assert!(always_save[1] <= sol.values[1] + 1e-9);
    }

    #[test]
    fn mdp_validation() {
        let mut bad = inventory_like();
        bad.transitions[0][0] = vec![(0, 0.4)];
        assert!(bad.validate().is_err());
        assert!(inventory_like().value_iteration(1.0, 1e-6, 10).is_err());
        assert!(inventory_like().backward_induction(3, &[0.0]).is_err());
        assert!(inventory_like().evaluate_policy(&[0, 5], 0.5, 1e-6).is_err());
        let empty = FiniteMdp { transitions: vec![], rewards: vec![] };
        assert!(empty.validate().is_err());
    }
}
