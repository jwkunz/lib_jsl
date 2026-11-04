"""Reference values for the chapter on graph algorithms, from SciPy's sparse.csgraph and NetworkX
(the book used SciPy 1.17.1 and NetworkX 3.7).

    python book/generator/data/make_graph_refs.py > book/generator/data/graph_reference.txt

Random graphs are written as `graph <name>|n;directed;u,v,w,u,v,w,...` lines, followed by the
results computed on them as `<case>|values`.
"""

import networkx as nx
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse import csgraph

rng = np.random.default_rng(2026)


def fmt(v):
    return "inf" if np.isinf(v) else repr(float(v))


def out(name, values):
    print(name + "|" + ",".join(fmt(v) for v in values))


def graph(name, n, m, directed, low, high, integer=False):
    pairs = set()
    edges = []
    while len(edges) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        key = (u, v) if directed else (min(u, v), max(u, v))
        if u == v or key in pairs:
            continue
        pairs.add(key)
        w = float(rng.integers(low, high + 1)) if integer else round(float(rng.uniform(low, high)), 3)
        edges.append((u, v, w))
    print(f"graph {name}|{n};{int(directed)};" + ",".join(f"{u},{v},{fmt(w)}" for u, v, w in edges))
    return edges


def matrix(n, edges, directed):
    rows, cols, vals = zip(*edges)
    a = csr_array((vals, (rows, cols)), shape=(n, n))
    return a


# Non-negative directed weights: Dijkstra and Floyd-Warshall from a few sources.
e = graph("positive", 200, 1200, True, 0.1, 10.0)
a = matrix(200, e, True)
for s in [0, 17, 199]:
    out(f"dijkstra positive {s}", csgraph.dijkstra(a, directed=True, indices=s))
fw = csgraph.floyd_warshall(a, directed=True)
out("floyd positive rows", np.concatenate([fw[3], fw[150]]))

# Negative weights by reweighting positive ones with vertex potentials, w + p(u) - p(v): every
# cycle keeps its positive total, so there are no negative cycles.
potential = np.round(rng.uniform(0.0, 20.0, 150), 3)
base = graph("signed_base", 150, 900, True, 0.1, 10.0)
e = [(u, v, round(w + potential[u] - potential[v], 3)) for (u, v, w) in base]
print("graph signed|150;1;" + ",".join(f"{u},{v},{fmt(w)}" for u, v, w in e))
out("bellman signed 0", csgraph.bellman_ford(matrix(150, e, True), directed=True, indices=0))

# Undirected: minimum spanning tree weight and connected components.
e = graph("undirected", 400, 700, False, 1.0, 100.0)
a = matrix(400, e, False)
mst = csgraph.minimum_spanning_tree(a)
out("mst undirected", [mst.sum()])
out("components undirected", [csgraph.connected_components(a, directed=False)[0]])

# Integer capacities: maximum flow from 0 to 99 by both algorithms.
e = graph("flow", 100, 700, True, 1, 50, integer=True)
a = csr_array(([int(w) for _, _, w in e], ([u for u, _, _ in e], [v for _, v, _ in e])), shape=(100, 100))
out("maxflow", [csgraph.maximum_flow(a, 0, 99, method="edmonds_karp").flow_value, csgraph.maximum_flow(a, 0, 99, method="dinic").flow_value])

# Sparse directed graph: strongly connected components.
e = graph("sparse", 300, 420, True, 1.0, 1.0)
k, labels = csgraph.connected_components(matrix(300, e, True), directed=True, connection="strong")
out("scc sparse", [k])
# Canonical partition: each vertex labelled by the smallest vertex in its component.
smallest = {}
for v, l in enumerate(labels):
    smallest.setdefault(l, v)
out("scc sparse partition", [smallest[l] for l in labels])

# A random DAG (edges from smaller to larger index): the lexicographically smallest topological order.
e = [(u, v, w) for (u, v, w) in graph("dag_raw", 60, 150, True, 1.0, 1.0) if u < v]
print("graph dag|60;1;" + ",".join(f"{u},{v},{fmt(w)}" for u, v, w in e))
d = nx.DiGraph()
d.add_nodes_from(range(60))
d.add_edges_from((u, v) for u, v, _ in e)
out("topological dag", list(nx.lexicographical_topological_sort(d)))
