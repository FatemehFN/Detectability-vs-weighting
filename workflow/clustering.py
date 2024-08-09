from infomap import Infomap
import igraph
import leidenalg
import numpy as np
from scipy import sparse


clustering_models = {}
clustering_model = lambda f: clustering_models.setdefault(f.__name__, f)


@clustering_model
def infomap(src, trg, weight):
    n_nodes = int(np.maximum(np.max(src), np.max(trg))) + 1

    im = Infomap("--two-level --directed")
    for i in range(len(src)):
        im.add_link(src[i], trg[i], weight[i])
    im.run()
    cids = np.zeros(n_nodes)
    for node in im.tree:
        if node.is_leaf:
            cids[node.node_id] = node.module_id
    return np.unique(cids, return_inverse=True)[1]


@clustering_model
def leiden(src, trg, weight):
    g = igraph.Graph.TupleList(
        [[src[i], trg[i], weight[i]] for i in range(len(src))],
        weights=True,
        directed=False,
    )
    weights = [edge["weight"] for edge in g.es]
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights=weights
    )
    n_nodes = int(np.maximum(np.max(src), np.max(trg))) + 1
    node_idx = np.array([g.vs[i]["name"] for i in range(len(g.vs))])
    memberships = np.zeros(n_nodes)
    for i, p in enumerate(part):
        memberships[node_idx[p]] = i
    return memberships
