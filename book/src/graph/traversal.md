# Graphs, traversal and orderings

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Store a graph, visit its vertices systematically, split it into connected or strongly connected components, and order a directed acyclic graph so that every edge points forward |
| **Devised by** | Leonhard Euler (1736), graph theory; Charles Trémaux (19th century) and Edward Moore (1959), depth- and breadth-first search; Arthur Kahn (1962), topological sorting; Robert Tarjan (1972), strongly connected components |
| **Accuracy** | Exact; strongly connected components and the lexicographically smallest topological order agree with SciPy and NetworkX (see the [chapter overview](index.md#testing-against-scipy-and-networkx)) |
| **Cost** | $O(V + E)$ for traversals and Tarjan's algorithm; $O((V + E)\log V)$ for the smallest topological order |
| **API** | [`Graph`](../../api/lib_jsl_graph/graph/struct.Graph.html) (`from_edges`, `from_unweighted_edges`, `from_adjacency_matrix`, `add_edge`, `neighbors`, `edges`, `adjacency_matrix`, `reversed`, `breadth_first`, `depth_first`, `connected_components`), [`Edge`](../../api/lib_jsl_graph/graph/struct.Edge.html), [`topological_sort`](../../api/lib_jsl_graph/ordering/fn.topological_sort.html), [`strongly_connected_components`](../../api/lib_jsl_graph/ordering/fn.strongly_connected_components.html), [`condensation`](../../api/lib_jsl_graph/ordering/fn.condensation.html) |

</div>

## Overview

A **graph** is a set of vertices joined by edges, which may be directed and may carry weights. It models road networks, dependencies, circuits and social ties. The library stores graphs as **adjacency lists**: for each vertex, the list of its out-neighbours with edge weights. This takes $O(V + E)$ memory and gives each vertex's neighbours directly. **Breadth-first search** visits vertices in order of distance (in edges) from a source, and **depth-first search** goes as deep as possible before backing up. Built on these are the **connected components** of an undirected graph and the **strongly connected components** of a directed one, the maximal sets of mutually reachable vertices.
A directed graph without cycles has a **topological order**, in which every edge goes from an earlier vertex to a later one. That is the order in which to compile modules, schedule tasks or evaluate a spreadsheet.

## History

Leonhard Euler's solution of the Königsberg bridge problem (1736) began graph theory. Charles Trémaux described depth-first exploration of mazes in the 19th century. Edward Moore (1959) used breadth-first search for shortest paths in mazes, and C. Y. Lee (1961) used it to route wires on circuit boards. Arthur Kahn (1962) gave the topological sort by repeatedly removing vertices with no incoming edges. Robert Tarjan's 1972 paper on depth-first search found strongly connected components in linear time, using one search and a stack. With John Hopcroft he showed how much a single depth-first search reveals, work for which they shared the 1986 Turing Award.

## The idea

### Representation

[`Graph`](../../api/lib_jsl_graph/graph/struct.Graph.html) holds vertices $0, \ldots, n - 1$ and, for each, a list of `(neighbour, weight)` pairs. An undirected edge is stored in both endpoints' lists and reported once by `edges`. Graphs can be built from edge lists, unweighted pairs or an adjacency matrix, and converted back. An adjacency matrix suits dense graphs and constant-time edge queries; lists suit the sparse graphs common in practice.

### Breadth-first and depth-first search

Breadth-first search keeps a queue. It takes the oldest vertex, and appends its unvisited neighbours. Vertices therefore come out in order of their edge distance from the source, and the first time a vertex is reached is by a shortest path in edges. Depth-first search keeps a stack instead. The library's version is iterative, so a path of a million vertices does not overflow the call stack. **Connected components** of an undirected graph (weakly connected for directed ones) are computed with a [union-find](spanning-trees.md#union-find) structure over the edges.

### Topological sorting

**Kahn's algorithm** counts each vertex's incoming edges and repeatedly removes a vertex with none left, decrementing its successors' counts. If vertices remain when none is free, they lie on a cycle and no topological order exists. The library always removes the smallest free vertex, using a min-heap, so the result is the **lexicographically smallest** topological order, a canonical answer when many exist. Writing this chapter fixed this: the original version used a first-in-first-out queue, which does not give the order it documented.

### Strongly connected components

Tarjan's algorithm numbers vertices in depth-first order, and tracks for each the lowest number reachable through its subtree and one back edge, its **low-link**. Vertices are pushed on a stack as they are discovered. When a vertex's low-link equals its own number, it is the root of a component, and the component is everything above it on the stack. Components come out in reverse topological order of the **condensation**, the acyclic graph obtained by shrinking each component to a vertex.

## Algorithm

**Breadth-first search** from a source.

1. Mark the source as seen and put it in a queue.
2. While the queue is not empty: remove the vertex at the front and record it; for each of its
   neighbours not yet seen, mark it as seen and add it to the back of the queue.

Marking a vertex when it is *added* to the queue, not when it is removed, ensures that it is
added only once. **Depth-first search** is the same with a stack in place of the queue. Both
cost $O(V + E)$.

**Topological sort** (Kahn's algorithm) of a directed graph.

1. Count the incoming edges of every vertex.
2. Put every vertex with no incoming edges in a min-heap.
3. While the heap is not empty: remove the smallest vertex and append it to the order; for
   each of its successors, reduce the count by one, and if the count reaches zero, add the
   successor to the heap.
4. If the order contains fewer than $V$ vertices, the remaining ones lie on a cycle: report
   an error.

**Strongly connected components** (Tarjan's algorithm). A counter numbers the vertices in
order of discovery. Each vertex has an **index** (its number) and a **low-link** (the smallest
index it is known to reach). Discovered vertices wait on a stack until their component is
complete. The depth-first search is run without recursion, with an explicit stack of frames,
each holding a vertex and the position of the next neighbour to examine. Start a search from
every vertex not yet discovered.

1. **Discover** a vertex: give it the next index, set its low-link to the same number, and
   push it on the component stack.
2. **Examine** its next neighbour. If the neighbour is undiscovered, descend into it (step 1).
   If it is still on the component stack, lower the vertex's low-link to the neighbour's index
   if that is smaller. If it has already been assigned to a component, ignore it.
3. **Finish** a vertex when its neighbours are exhausted. If its low-link equals its own
   index, it is the root of a component: pop the component stack down to and including this
   vertex, and those vertices form one component. Then return to the parent frame and lower
   the parent's low-link to this vertex's low-link if that is smaller.

## Implementation

```rust,ignore
{{#include ../../../crates/graph/src/ordering.rs:tarjan}}
```

- Vertices are `usize` indices; weights are `f64` (1 for unweighted edges).
- `condensation` returns the component labels and the component DAG, with one edge per connected pair of components.

## In practice

```rust,ignore
{{#include ../../../crates/graph/examples/graph_traversal.rs}}
```

```text
breadth-first from 0: [0, 1, 2, 3, 4]
depth-first from 0:   [0, 1, 3, 2, 4]
components: [0, 0, 0, 0, 0, 1, 1]
a study order: [0, 1, 2, 3, 4, 5]
strongly connected components: [2, 2, 2, 1, 1, 0]
condensation: 3 components, edges [(1, 0), (2, 1)]
```

The sink component $\{5\}$ is numbered first and the source component $\{0, 1, 2\}$ last: the labels are a reverse topological order.

**Strengths**

- Linear-time traversals and components; iterative, so deep graphs are safe.
- A canonical topological order and a condensation for dependency analysis.

**Weaknesses**

- Vertices must be numbered $0, \ldots, n - 1$; no labels, deletion or dynamic updates.
- Adjacency lists make edge-existence queries $O(\deg v)$.

## Exercises

1. Show that breadth-first search reaches each vertex first along a path with the fewest edges.
2. Prove that Kahn's algorithm finishes with every vertex exactly when the graph is acyclic.
3. Show that the lexicographically smallest topological order is obtained by always removing the smallest free vertex.
4. Trace Tarjan's algorithm on the example's directed graph and check the low-link values.
5. Show that the condensation of any directed graph is acyclic.

## Further reading

- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022, chapter 20.
- R. E. Tarjan, "Depth-first search and linear graph algorithms", *SIAM Journal on Computing* 1 (1972) 146–160.
- A. B. Kahn, "Topological sorting of large networks", *Communications of the ACM* 5 (1962) 558–562.
- R. Sedgewick and K. Wayne, *Algorithms*, 4th ed., Addison-Wesley, 2011, chapter 4.
