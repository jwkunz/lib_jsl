# Dynamic programming

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Optimise a sequence of decisions, or a combinatorial structure that decomposes into overlapping subproblems: knapsack, coin change, sequence alignment, matrix-chain ordering, Markov decision processes |
| **Devised by** | Richard Bellman (1950s); edit distance: Vladimir Levenshtein (1965), Robert Wagner and Michael Fischer (1974); Markov decision processes: Bellman (1957), Ronald Howard (1960) |
| **Accuracy** | Exact for the discrete problems; value iteration converges geometrically, with error $O(\gamma^k)$ after $k$ iterations for discount $\gamma$ |
| **Cost** | The number of subproblems times the work per subproblem: $O(nW)$ for knapsack, $O(nm)$ for edit distance, $O(n^3)$ for matrix chains, $O(|S|^2|A|)$ per value iteration |
| **API** | [`knapsack_01`](../../api/lib_jsl_optimization/dynamic_programming/fn.knapsack_01.html), [`coin_change_min_coins`](../../api/lib_jsl_optimization/dynamic_programming/fn.coin_change_min_coins.html), [`edit_distance`](../../api/lib_jsl_optimization/dynamic_programming/fn.edit_distance.html), [`longest_common_subsequence`](../../api/lib_jsl_optimization/dynamic_programming/fn.longest_common_subsequence.html), [`longest_increasing_subsequence`](../../api/lib_jsl_optimization/dynamic_programming/fn.longest_increasing_subsequence.html), [`matrix_chain_order`](../../api/lib_jsl_optimization/dynamic_programming/fn.matrix_chain_order.html), [`FiniteMdp`](../../api/lib_jsl_optimization/dynamic_programming/struct.FiniteMdp.html) |

</div>

## Overview

Many optimisation problems have a recursive structure: the best solution of the whole is built
from the best solutions of parts, and the same parts recur many times. **Dynamic programming**
solves each subproblem once, stores its answer in a table, and builds up to the full problem. Where
naive recursion or exhaustive search takes exponential time, dynamic programming often takes
polynomial time.

The idea applies far beyond the examples here: shortest paths in graphs (Bellman-Ford), the
Viterbi algorithm for hidden Markov models, sequence alignment in bioinformatics, parsing, optimal
control and reinforcement learning all are dynamic programming.

## History

Richard Bellman developed dynamic programming at the RAND Corporation in the early 1950s, for
multistage decision problems in logistics and control. He later wrote that he chose the name to
shield the mathematical research from a Secretary of Defense who disliked the word "research":
"programming" meant planning, and "dynamic" sounded impressive and could not be used pejoratively.
His book *Dynamic Programming* (1957) stated the **principle of optimality** and introduced Markov
decision processes, for which Ronald Howard (1960) developed policy iteration.

Many of the classic algorithms are dynamic programs found independently in different fields:
Vladimir Levenshtein defined the edit distance in 1965 for error-correcting codes, Saul Needleman and
Christian Wunsch (1970) aligned protein sequences, and Robert Wagner and Michael Fischer (1974)
published the general string-to-string algorithm. Andrew Viterbi's decoding algorithm (1967) and the
matrix-chain algorithm (Godbole, 1973) followed the same pattern.

## The idea

### Bellman's principle

> **Principle of optimality.** An optimal policy has the property that, whatever the initial state
> and initial decision, the remaining decisions must constitute an optimal policy with regard to the
> state resulting from the first decision.

The principle turns into a recurrence (the **Bellman equation**) for the optimal value of each
subproblem in terms of smaller ones. Two ingredients make the recurrence efficient: **optimal
substructure** (the principle holds) and **overlapping subproblems** (the same subproblems recur, so
storing their values pays). The values are computed bottom-up into a table, and an optimal solution
is recovered by tracing back the choices that achieved them.

### Sequence and combinatorial problems

- **0/1 knapsack**: with $V(i, w)$ the best value from the first $i$ items within capacity $w$,
  $V(i, w) = \max(V(i-1, w),\ V(i-1, w - w_i) + v_i)$. The table has $(n + 1)(W + 1)$ entries: polynomial in
  the capacity $W$ but exponential in its number of digits (the problem is NP-hard).
- **Coin change**: the fewest coins for amount $a$ is $1 + \min_c C(a - c)$. Unlike the greedy method,
  which fails for coins $\{1, 3, 4\}$ and amount 6, it is always optimal.
- **Edit distance**: with $D(i, j)$ the distance between the first $i$ and $j$ characters,
  $D(i, j) = \min(D(i-1, j) + 1,\ D(i, j-1) + 1,\ D(i-1, j-1) + [a_i \ne b_j])$.
- **Longest common subsequence**: the same table with matches scoring 1 and no substitutions.
- **Longest increasing subsequence**: an $O(n \log n)$ refinement that keeps, for each length, the
  smallest possible last element ("patience sorting").
- **Matrix-chain ordering**: the cheapest parenthesisation of $A_1A_2\cdots A_n$, from the costs of all
  subchains, $O(n^3)$.

{{#include ../generated/optimization/dynamic_programming.md}}

### Markov decision processes

A **Markov decision process** has states $s$, actions $a$, rewards $r(s, a)$ and transition probabilities
$P(s' \mid s, a)$. A **policy** chooses an action in each state, and its value is the expected discounted
sum of rewards, $\mathbb E\sum_t\gamma^t r_t$, with discount $0 \le \gamma < 1$. The optimal values satisfy the Bellman
equation

$$
V^*(s) = \max_a\Big[r(s, a) + \gamma\sum_{s'}P(s' \mid s, a)\,V^*(s')\Big],
$$

and the optimal policy takes the maximising action. **Value iteration** applies the right-hand side
repeatedly, starting from $V = 0$. The map is a **contraction** with factor $\gamma$ in the maximum norm, so
the error falls like $\gamma^k$:

![Value iteration](../generated/optimization/value_iteration.svg)

With $\gamma = 0.99$ (rewards a hundred steps ahead still matter), thousands of iterations are needed;
policy iteration, which solves for the value of the current policy exactly at each step, converges
in far fewer, more expensive, iterations. For a finite horizon $T$, **backward induction** computes the
optimal values stage by stage from the end, exactly, in $T$ steps.

## Algorithm

For each problem: allocate the table, fill it in an order in which every entry's dependencies come
first, and trace back from the final entry to recover a solution.

**Value iteration**: from $V = 0$, repeat $V \leftarrow \max_a[r + \gamma PV]$ until the largest change is below
$\text{tol}\,(1 - \gamma)/(2\gamma)$, which guarantees that the greedy policy is within the tolerance of
optimal.

## Implementation

```rust,ignore
{{#include ../../../crates/optimization/src/dynamic_programming.rs:knapsack}}
```

```rust,ignore
{{#include ../../../crates/optimization/src/dynamic_programming.rs:value_iteration}}
```

- Each sequence function returns the optimal value together with one optimal solution (chosen
  items, coin counts, matched index pairs, an edit script, a parenthesisation).
- `FiniteMdp::validate` checks that the transition probabilities of every action sum to 1;
  `backward_induction`, `value_iteration` and `evaluate_policy` solve finite-horizon, discounted and
  fixed-policy problems.
- The tables are dense: knapsack needs $(n + 1)(W + 1)$ numbers, so very large capacities are
  impractical.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_dynamic_programming.rs}}
```

**Strengths**

- Exact optimal solutions to problems that look exponential.
- A single principle covers sequence alignment, scheduling, control and decision-making under
  uncertainty.

**Weaknesses**

- The table can be enormous (the "curse of dimensionality", Bellman's own phrase), which limits
  dynamic programming to problems with a small state space.
- Each problem needs its own formulation; finding the right subproblems is the hard part.

## Exercises

1. Show that the greedy method (largest coin first) is optimal for coins $\{1, 5, 10, 25\}$ but not for
   $\{1, 3, 4\}$.
2. Modify the edit distance to give substitutions cost 2. What does the distance then count?
3. Count the parenthesisations of a chain of $n$ matrices (the Catalan numbers) and compare with the
   $O(n^3)$ cost of the dynamic program.
4. Prove that the Bellman operator is a contraction with factor $\gamma$ in the maximum norm.
5. Implement policy iteration for `FiniteMdp` using `evaluate_policy`, and compare its iterations with
   value iteration for $\gamma = 0.99$.

## Further reading

- R. Bellman, *Dynamic Programming*, Princeton University Press, 1957 (Dover reprint, 2003).
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed., MIT
  Press, 2022, chapter 14.
- R. A. Wagner and M. J. Fischer, "The string-to-string correction problem", *Journal of the ACM* 21
  (1974) 168–173.
- M. L. Puterman, *Markov Decision Processes*, Wiley, 1994.
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018,
  chapter 4.
