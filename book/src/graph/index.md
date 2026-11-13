# Graph algorithms: traversal, shortest paths, spanning trees and flows

Graphs model anything made of things and the connections between them: roads, circuits, dependencies, networks of computers or people. This chapter covers `lib_jsl_graph`:
- traversal and orderings (breadth- and depth-first search, components, topological order, strongly connected components);
- shortest paths, from one source or between all pairs;
- minimum spanning trees and the union-find structure;
- maximum flow with its dual, the minimum cut.

## The central questions

- **Directed or undirected, weighted or not?** The same adjacency-list type holds all four kinds; each algorithm states which it needs (Dijkstra non-negative weights, spanning trees undirected graphs, topological sorting a directed acyclic graph).
- **One target, one source or all pairs?** A* for one target, Dijkstra or Bellman–Ford for one source, Floyd–Warshall for all pairs.
- **Greedy or iterative?** Dijkstra, Kruskal and Prim are greedy and correct for structural reasons; Bellman–Ford and the flow algorithms improve a solution until it cannot be improved.

## Testing against SciPy and NetworkX

The references come from SciPy 1.17.1's `sparse.csgraph` and NetworkX 3.7, on random graphs written out by `book/generator/data/make_graph_refs.py` and rebuilt identically by the generator:

<p class="caption" id="tab-23-1"><strong>Table 23.1.</strong> Columns: computation (reference function), result.</p>

{{#include ../generated/graph/reference.md}}

## In the library

All of these are in `lib_jsl_graph` ([API reference](../../api/lib_jsl_graph/index.html)).

| You need | Use | Entry |
|---|---|---|
| A graph structure; traversal; components; an order for dependencies | `Graph`, `breadth_first`, `depth_first`, `connected_components`, `topological_sort`, `strongly_connected_components`, `condensation` | [Traversal and orderings](traversal.md) |
| Cheapest paths | `dijkstra`, `bellman_ford`, `a_star`, `floyd_warshall` | [Shortest paths](shortest-paths.md) |
| The cheapest connected network; disjoint sets | `kruskal`, `prim`, `UnionFind` | [Spanning trees](spanning-trees.md) |
| The most flow through a network; its bottleneck | `FlowNetwork::edmonds_karp`, `FlowNetwork::dinic` | [Maximum flow](max-flow.md) |
