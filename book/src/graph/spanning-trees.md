# Minimum spanning trees and union-find

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Connect all the vertices of a weighted undirected graph with the least total edge weight, and keep track of which vertices are already connected |
| **Devised by** | Otakar Borůvka (1926); Joseph Kruskal (1956); Vojtěch Jarník (1930), Robert Prim (1957) and Edsger Dijkstra (1959); Bernard Galler and Michael Fischer (1964) and Robert Tarjan (1975), union-find |
| **Accuracy** | Exact; Kruskal and Prim agree with SciPy's `minimum_spanning_tree` (see the [chapter overview](index.md#testing-against-scipy-and-networkx)) |
| **Cost** | Kruskal $O(E\log E)$; Prim $O(E\log V)$; union-find operations in near-constant amortised time, $O(\alpha(n))$ |
| **API** | [`kruskal`](../../api/lib_jsl_graph/spanning_trees/fn.kruskal.html), [`prim`](../../api/lib_jsl_graph/spanning_trees/fn.prim.html), [`SpanningForest`](../../api/lib_jsl_graph/spanning_trees/struct.SpanningForest.html), [`UnionFind`](../../api/lib_jsl_graph/graph/struct.UnionFind.html) (`find`, `union`, `connected`, `set_count`) |

</div>

## Overview

A **spanning tree** of a connected graph is a set of $V - 1$ edges that connects every vertex without cycles. A **minimum spanning tree** (MST) has the smallest total weight. It is the cheapest way to wire a network, and appears inside clustering ([single linkage](../unsupervised/hierarchical.md) is an MST cut into pieces), approximation algorithms and image segmentation. Two greedy algorithms find it. **Kruskal's** adds edges from cheapest to dearest, skipping any that would close a cycle. **Prim's** grows one tree from a root by always adding the cheapest edge leaving it.
Kruskal's needs to know quickly whether two vertices are already connected, which is the job of the **union-find** (disjoint-set) structure. For a disconnected graph both return a minimum spanning **forest**, one tree per component.

## History

Otakar Borůvka (1926) solved the problem to plan Moravia's electrical network. Vojtěch Jarník (1930) gave the vertex-growing algorithm, rediscovered by Robert Prim (1957) at Bell Labs and by Edsger Dijkstra (1959). Joseph Kruskal (1956) published the edge-sorting algorithm while a graduate student. The union-find structure was introduced by Bernard Galler and Michael Fischer (1964). Robert Tarjan (1975) proved that with union by rank and path compression its operations cost $O(\alpha(n))$ amortised, where $\alpha$, the inverse Ackermann function, is below 5 for any conceivable $n$.

## The idea

### The cut property

For any division of the vertices into two sets, the cheapest edge crossing between them belongs to some minimum spanning tree. Both greedy algorithms only ever add such an edge, so both are correct. If the weights are distinct, the MST is unique.

### Kruskal

Sort the edges by weight. Take each in turn, and keep it if its endpoints are in different components, merging them. An edge whose endpoints are already connected would close a cycle, and is the most expensive edge on that cycle, so it can be skipped.

### Prim

Start from a root. A heap holds the edges leaving the tree. Pop the cheapest; if it reaches a new vertex, add it and push that vertex's edges. Stale edges, to vertices already in the tree, are skipped. On dense graphs Prim is preferred, since Kruskal must sort every edge.

![Euclidean MST](../generated/graph/euclidean_mst.svg)

{{#include ../generated/graph/mst_facts.md}}

### Union-find

Each set is a tree of parent pointers whose root represents the set. `find` follows parents to the root, halving the path as it goes (each visited node is pointed to its grandparent). `union` hangs the root of lower **rank** (an upper bound on height) under the other. Together these keep the trees almost flat.

## Algorithm

1. **Kruskal**: sort edges; for each, if `union(u, v)` succeeds (different sets), keep it.
2. **Prim**: for each vertex not yet in a tree, start a tree there; heap of leaving edges; pop, skip if stale, add and push the new vertex's edges.
3. **Find with path halving**: while $x$ is not its own parent, set parent$(x)$ to its grandparent and move $x$ to it.

## Implementation

```rust,ignore
{{#include ../../../crates/graph/src/spanning_trees.rs:kruskal}}
```

```rust,ignore
{{#include ../../../crates/graph/src/graph.rs:union_find}}
```

- Both algorithms require an undirected graph and return the forest's edges and total weight.
- `Graph::connected_components` uses the same union-find structure.

## In practice

```rust,ignore
{{#include ../../../crates/graph/examples/graph_spanning_trees.rs}}
```

```text
Kruskal: weight 37, edges [(6, 7), (2, 8), (5, 6), (0, 1), (2, 5), (2, 3), (0, 7), (3, 4)]
Prim:    weight 37, 8 edges
union-find: 0 and 2 connected true, 0 and 5 connected false, 3 sets
```

Kruskal lists its edges in the order it accepts them, cheapest first.

**Strengths**

- Simple, fast greedy algorithms; minimum spanning forests for disconnected graphs.
- A union-find structure useful well beyond spanning trees.

**Weaknesses**

- No Borůvka (parallel-friendly) variant, and no maximum or degree-constrained trees.
- Kruskal sorts all edges even when the graph is dense.

## Exercises

1. Prove the cut property, and deduce the correctness of Kruskal's and Prim's algorithms.
2. Show that with distinct weights the minimum spanning tree is unique.
3. Show that the heaviest edge of any cycle is in no minimum spanning tree (with distinct weights).
4. Show that single-linkage clustering into $k$ clusters is the MST with its $k - 1$ heaviest edges removed.
5. Show that union by rank alone keeps tree heights at most $\log_2 n$.

## Further reading

- J. B. Kruskal, "On the shortest spanning subtree of a graph and the traveling salesman problem", *Proceedings of the American Mathematical Society* 7 (1956) 48–50.
- R. C. Prim, "Shortest connection networks and some generalizations", *Bell System Technical Journal* 36 (1957) 1389–1401.
- R. E. Tarjan, "Efficiency of a good but not linear set union algorithm", *Journal of the ACM* 22 (1975) 215–225.
- R. L. Graham and P. Hell, "On the history of the minimum spanning tree problem", *Annals of the History of Computing* 7 (1985) 43–57.
