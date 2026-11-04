# Maximum flow and minimum cut: Edmonds–Karp and Dinic

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Send as much flow as possible from a source to a sink through a network of capacitated arcs, and find the cheapest set of arcs whose removal separates them |
| **Devised by** | Lester Ford and Delbert Fulkerson (1956), augmenting paths and the max-flow min-cut theorem; Jack Edmonds and Richard Karp (1972), shortest augmenting paths; Yefim Dinitz (1970), blocking flows in level graphs |
| **Accuracy** | Exact for integer capacities (to rounding for real ones); both algorithms agree with SciPy's `maximum_flow`, and the cut found always equals the flow (tables below) |
| **Cost** | Edmonds–Karp $O(VE^2)$; Dinic $O(V^2E)$, and $O(E\sqrt V)$ on unit-capacity networks |
| **API** | [`FlowNetwork`](../../api/lib_jsl_graph/max_flow/struct.FlowNetwork.html) (`new`, `add_edge`, `edmonds_karp`, `dinic`), [`MaxFlow`](../../api/lib_jsl_graph/max_flow/struct.MaxFlow.html) (`value`, `edge_flows`, `source_side`) |

</div>

## Overview

In a **flow network** each arc has a capacity. A **flow** assigns each arc an amount up to its capacity, such that flow is conserved at every vertex except the source and the sink. The **maximum flow** problem asks how much can get from source to sink. Its dual is the **minimum cut**: a division of the vertices into a source side and a sink side, minimising the capacity of the arcs crossing from one to the other. The **max-flow min-cut theorem** says the two values are equal. The algorithms push flow along **augmenting paths** in the **residual network**, which records how much more each arc can carry and how much existing flow can be cancelled. Edmonds–Karp augments along shortest paths. Dinic augments a whole layer of shortest paths at once.
Applications range from transport and bipartite matching to image segmentation and baseball elimination.

## History

Lester Ford and Delbert Fulkerson (1956) proved the max-flow min-cut theorem and gave the augmenting-path method. The problem had been posed by Theodore Harris and Frank Ross (1955) to study the capacity of the Soviet railway network. With arbitrary augmenting paths the method can take exponentially many steps, or fail to terminate with irrational capacities. Jack Edmonds and Richard Karp (1972) showed that shortest paths bound it polynomially. Independently, Yefim Dinitz (1970) in Moscow introduced level graphs and blocking flows, faster still. Push-relabel methods (Goldberg and Tarjan, 1988) and, in 2022, an almost-linear-time algorithm (Chen, Kyng, Liu, Peng, Probst Gutenberg and Sachdeva) followed.

## The idea

### Residual networks and augmenting paths

For each arc $u \to v$ with capacity $c$ and flow $f$, the residual network has a forward arc of residual capacity $c - f$ and a backward arc of capacity $f$, which can cancel flow. Any source-to-sink path of positive residual capacity can carry its bottleneck amount more. When no such path remains, the vertices reachable from the source in the residual network form a cut whose arcs are all saturated, so the flow equals the cut capacity and both are optimal.

### Edmonds–Karp

Choosing the augmenting path by breadth-first search, so it has the fewest arcs, guarantees that the distance from source to sink never decreases. Each arc can be the bottleneck only $O(V)$ times. Hence at most $O(VE)$ augmentations, each $O(E)$.

### Dinic

Breadth-first search labels each vertex with its distance from the source. The **level graph** keeps only arcs from one level to the next. A **blocking flow** in it saturates at least one arc of every shortest path. It is found by depth-first searches that remember, for each vertex, which arc to try next (the current-arc pointer), and retreat past dead ends. After each phase the source-to-sink distance strictly increases, so there are at most $V$ phases.

{{#include ../generated/graph/flow_table.md}}

In every case the minimum cut read from the final residual network has the same capacity as the flow, as the theorem says. Dinic is up to four times faster here.

## Algorithm

1. **Residual arcs**: each `add_edge` stores a forward arc and a reverse arc of capacity zero at indices $2k$ and $2k + 1$, so `arc ^ 1` is the partner.
2. **Edmonds–Karp**: BFS for a path with residual capacity; push the bottleneck along it; repeat until the sink is unreachable.
3. **Dinic**: BFS levels; repeated DFS along level-increasing arcs with current-arc pointers, pushing each path's bottleneck; repeat phases until the sink is unreachable.
4. **Minimum cut**: BFS from the source over arcs with residual capacity; the reached vertices are the source side.

## Implementation

```rust,ignore
{{#include ../../../crates/graph/src/max_flow.rs:dinic_phase}}
```

- Capacities are `f64`; residual capacities below $10^{-12}$ count as zero.
- Each call resets the flow, so one network can be solved with both algorithms or other terminals.

## In practice

```rust,ignore
{{#include ../../../crates/graph/examples/graph_max_flow.rs}}
```

```text
maximum flow: Edmonds-Karp 23, Dinic 23
arc flows: [12.0, 11.0, 0.0, 0.0, 12.0, 0.0, 11.0, 7.0, 19.0, 4.0]
minimum cut [(1, 3, 12.0), (4, 3, 7.0), (4, 5, 4.0)], capacity 23
```

This is the classic textbook network: three saturated arcs, of capacities 12, 7 and 4, form the bottleneck.

**Strengths**

- Two standard algorithms with flows per arc and a minimum cut.
- Dinic is fast on the sparse networks of matching and segmentation problems.

**Weaknesses**

- One source and one sink (add a super-source for several); no costs (min-cost flow) or lower bounds.
- Real capacities rely on a tolerance; integer problems are best given integer-valued capacities.

## Exercises

1. Prove the max-flow min-cut theorem from the residual network at termination.
2. Find a network where choosing augmenting paths badly takes as many steps as the flow value.
3. Show that in Edmonds–Karp the BFS distance from the source to each vertex never decreases.
4. Reduce maximum bipartite matching to maximum flow, and solve a small example.
5. Show that each Dinic phase increases the source-to-sink distance in the residual network.

## Further reading

- L. R. Ford and D. R. Fulkerson, "Maximal flow through a network", *Canadian Journal of Mathematics* 8 (1956) 399–404.
- J. Edmonds and R. M. Karp, "Theoretical improvements in algorithmic efficiency for network flow problems", *Journal of the ACM* 19 (1972) 248–264.
- Y. Dinitz, "Dinitz' algorithm: the original version and Even's version", in *Theoretical Computer Science*, LNCS 3895, Springer, 2006, 218–240.
- R. K. Ahuja, T. L. Magnanti and J. B. Orlin, *Network Flows*, Prentice Hall, 1993.
