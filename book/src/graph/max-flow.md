# Maximum flow and minimum cut: Edmonds–Karp and Dinic

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Send as much flow as possible from a source to a sink through a network of capacitated arcs, and find the cheapest set of arcs whose removal separates them |
| **Devised by** | Lester Ford and Delbert Fulkerson (1956), augmenting paths and the max-flow min-cut theorem; Jack Edmonds and Richard Karp (1972), shortest augmenting paths; Yefim Dinitz (1970), blocking flows in level graphs |
| **Accuracy** | Exact for integer capacities (to rounding for real ones); both algorithms agree with SciPy's `maximum_flow`, and the cut found always equals the flow (tables below) |
| **Cost** | Edmonds–Karp $O(VE^2)$; Dinic $O(V^2E)$, and $O(E\sqrt V)$ on unit-capacity networks |
| **Read first** | [Shortest paths: Dijkstra, Bellman–Ford, A* and Floyd–Warshall](shortest-paths.md) |
| **API** | [`FlowNetwork`](../../api/lib_jsl_graph/max_flow/struct.FlowNetwork.html) (`new`, `add_edge`, `edmonds_karp`, `dinic`), [`MaxFlow`](../../api/lib_jsl_graph/max_flow/struct.MaxFlow.html) (`value`, `edge_flows`, `source_side`) |

</div>

## Overview

In a **flow network** each arc has a capacity. A **flow** assigns each arc an amount up to its capacity, such that flow is conserved at every vertex except the source and the sink. The **maximum flow** problem asks how much can get from source to sink. Its dual is the **minimum cut**: a division of the vertices into a source side and a sink side, minimising the capacity of the arcs crossing from one to the other. The **max-flow min-cut theorem** says the two values are equal. The algorithms push flow along **augmenting paths** in the **residual network**, which records how much more each arc can carry and how much existing flow can be cancelled. Edmonds–Karp augments along shortest paths. Dinic augments a whole layer of shortest paths at once.
Applications range from transport and bipartite matching to image segmentation and baseball elimination.

## History

Lester Ford and Delbert Fulkerson (1956) proved the max-flow min-cut theorem and gave the augmenting-path method. The problem had been posed by Theodore Harris and Frank Ross (1955) to study the capacity of the Soviet railway network. With arbitrary augmenting paths the method can take exponentially many steps, or fail to terminate with irrational capacities. Jack Edmonds and Richard Karp (1972) showed that shortest paths bound it polynomially. Independently, Yefim Dinitz (1970) in Moscow introduced level graphs and blocking flows, faster still. Push-relabel methods (Goldberg and Tarjan, 1988) and, in 2022, an almost-linear-time algorithm (Chen, Kyng, Liu, Peng, Probst Gutenberg and Sachdeva) followed.

## The idea

### Residual networks and augmenting paths

**The problem.** A **flow network** is a directed graph in which each arc has a **capacity**,
like a pipe that can carry so many litres per second. A **flow** assigns to each arc an amount
no greater than its capacity, such that at every vertex other than the **source** and the
**sink**, what flows in flows out. The task is to send as much as possible from the source to
the sink.

**Augmenting paths.** The natural approach is to find a path from source to sink along arcs
that still have spare capacity, push as much flow along it as its tightest arc (the
**bottleneck**) allows, and repeat. On its own this can get stuck with a poor answer, because an
early path may have used an arc that a better solution needs for something else. The remedy is
to allow later paths to *undo* earlier decisions.

That is the purpose of the **residual network**. For each arc $u \to v$ with capacity $c$
carrying flow $f$, it has

- a forward arc $u \to v$ of residual capacity $c - f$, the room still left, and
- a backward arc $v \to u$ of residual capacity $f$, which represents the option of cancelling
  flow already sent.

A path from source to sink in the residual network is an **augmenting path**. Pushing flow along
a backward arc reduces the flow on the original arc, rerouting it. Any augmenting path of
positive residual capacity can carry its bottleneck amount more, so the flow is not yet maximal.

**When to stop: the max-flow min-cut theorem.** A **cut** is a division of the vertices into
two sets, one containing the source and the other the sink; its capacity is the total capacity
of the arcs leading from the source's side to the sink's. Every unit of flow must cross the cut,
so no flow can exceed the capacity of any cut. Now suppose no augmenting path remains. Let $S$ be
the set of vertices still reachable from the source in the residual network; the sink is not
among them. Every arc leaving $S$ must be full (or it would extend the reachable set), and every
arc entering $S$ must be empty (or its backward arc would). So the flow across this cut equals
the cut's capacity. A flow equal to the capacity of some cut cannot be exceeded, so the flow is
maximal, and the cut is a minimum cut. The maximum flow equals the minimum cut.

### Edmonds–Karp

The method above does not say *which* augmenting path to choose, and a bad choice can take a
very long time. **Edmonds-Karp** always chooses a path with the fewest arcs, found by
[breadth-first search](traversal.md#breadth-first-and-depth-first-search). With that rule, the
distance from the source to any vertex in the residual network never decreases, and it follows
that each arc can be the bottleneck only $O(V)$ times. Hence there are at most $O(VE)$
augmentations, each costing one search, $O(E)$: $O(VE^2)$ in all, whatever the capacities.

### Dinic

Edmonds-Karp runs a fresh search for every path. Dinic's algorithm finds many shortest paths per
search. It works in phases.

1. Breadth-first search labels each vertex with its distance from the source in the residual
   network. The **level graph** keeps only the arcs that lead from one level to the next, so
   every path in it is a shortest path.
2. Find a **blocking flow** in the level graph: push flow along its paths until every path from
   source to sink contains a saturated arc. This is done by depth-first searches that remember,
   for each vertex, which arc to try next (the **current-arc pointer**), so that an arc found to
   be a dead end is never tried again in the same phase.

After each phase the shortest source-to-sink distance in the residual network strictly
increases, so there are at most $V$ phases, and the total cost is $O(V^2E)$.

<p class="caption" id="tab-23-2"><strong>Table 23.2.</strong> Maximum flow by Edmonds-Karp and Dinic on random networks.</p>

{{#include ../generated/graph/flow_table.md}}

In every case the minimum cut read from the final residual network has the same capacity as the flow, as the theorem says. Dinic is up to four times faster here.

## Algorithm

**Storing the network.** Each call to `add_edge` stores two arcs: the forward arc with the
given capacity, and a reverse arc of capacity zero. They are placed at consecutive positions
$2k$ and $2k + 1$ of one array, so that the partner of any arc is found by flipping the lowest
bit of its index (`arc ^ 1`). Pushing an amount of flow along an arc subtracts it from that
arc's residual capacity and adds it to the partner's.

**Edmonds-Karp.**

1. Run a breadth-first search from the source, following only arcs with positive residual
   capacity and recording the arc by which each vertex was reached.
2. If the sink was not reached, stop.
3. Walk back from the sink to the source along the recorded arcs to find the smallest
   residual capacity on the path, the bottleneck.
4. Walk back again, pushing the bottleneck amount along every arc of the path. Add it to the
   total flow, and go to 1.

Each augmentation costs $O(E)$, and there are at most $O(VE)$ of them.

**Dinic.**

1. Run a breadth-first search from the source over arcs with positive residual capacity,
   recording each vertex's **level**, its distance from the source. If the sink was not
   reached, stop.
2. Reset every vertex's current-arc pointer to its first arc.
3. Run a depth-first search from the source that follows only arcs with positive residual
   capacity leading from one level to the next. When the search finds that an arc leads
   nowhere, it advances the vertex's pointer past it for good. When it reaches the sink, push
   the bottleneck of the path found.
4. Repeat step 3 until the search finds no path, which completes one phase; then go to 1.

There are at most $V$ phases, each costing $O(VE)$.

**Minimum cut**, after either algorithm: run a breadth-first search from the source over arcs
with positive residual capacity. The vertices reached form the source side, and the original
arcs leading from them to the other vertices are the cut.

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

<details>
<summary>Hints and answers</summary>

1. At termination there is no augmenting path, so the set $S$ of vertices reachable from the source in the residual network does not contain the sink. Every edge leaving $S$ is saturated and every edge entering it carries zero flow, so the flow value equals the capacity of the cut $(S, V\setminus S)$; no flow can exceed any cut's capacity, so this cut is minimum.
2. Take edges $s \to a$, $s \to b$, $a \to t$ and $b \to t$ of capacity $M$ and an edge $a \to b$ of capacity 1. If the augmenting paths alternate between $s, a, b, t$ and $s, b, a, t$ (the second using the reverse of the middle edge), each pushes only 1 unit, so $2M$ augmentations are needed for a flow of $2M$.
3. Each augmentation saturates a bottleneck edge and can add only reverse edges that point from a farther to a nearer vertex; shortest-path distances in the residual network therefore never decrease.
4. Add a source joined to every left vertex and a sink joined from every right vertex, all capacities 1; the maximum flow equals the size of a maximum matching, and the saturated middle edges form the matching.
5. A phase saturates the level graph, after which no source-to-sink path in the level graph remains; any new path must use a reverse edge, which goes from a deeper to a shallower level, so it is strictly longer.

</details>

## Further reading

- L. R. Ford and D. R. Fulkerson, "Maximal flow through a network", *Canadian Journal of Mathematics* 8 (1956) 399–404.
- J. Edmonds and R. M. Karp, "Theoretical improvements in algorithmic efficiency for network flow problems", *Journal of the ACM* 19 (1972) 248–264.
- Y. Dinitz, "Dinitz' algorithm: the original version and Even's version", in *Theoretical Computer Science*, LNCS 3895, Springer, 2006, 218–240.
- R. K. Ahuja, T. L. Magnanti and J. B. Orlin, *Network Flows*, Prentice Hall, 1993.
