| computation (reference function) | result |
|---|---:|
| Dijkstra, 200 vertices, 1200 arcs, from three sources (`csgraph.dijkstra`) | largest difference 0.0e0 |
| Floyd-Warshall, same graph, two rows (`csgraph.floyd_warshall`) | largest difference 0.0e0 |
| Bellman-Ford with negative weights, 150 vertices (`csgraph.bellman_ford`) | largest difference 0.0e0 |
| minimum spanning forest weight, 400 vertices (`csgraph.minimum_spanning_tree`) | Kruskal 12987.185, Prim 12987.185, SciPy 12987.185 |
| connected components (`csgraph.connected_components`) | 14 (SciPy 14) |
| maximum flow, 100 vertices, 700 arcs (`csgraph.maximum_flow`) | Edmonds-Karp 193, Dinic 193, SciPy 193 |
| strongly connected components, 300 vertices (`connected_components(connection="strong")`) | 243 components (SciPy 243); same partition: true |
| lexicographically smallest topological order, 60 vertices (NetworkX) | identical: true |
