import infomap
import igraph
import leidenalg
import numpy as np
from scipy import sparse


clustering_models = {}
clustering_models = lambda f: clustering_models.setdefault(f.__name__, f)


@clustering_models
def detect_by_infomap(src, trg, weight):
    n_nodes = int(np.max(np.max(src), np.max(trg))) + 1

    im = infomap.Infomap("--two-level --directed")
    for i in range(len(src)):
        im.add_link(src[i], trg[i], weight[i])
    im.run()
    cids = np.zeros(n_nodes)
    for node in im.tree:
        if node.is_leaf:
            cids[node.node_id] = node.module_id
    return np.unique(cids, return_inverse=True)[1]


@clustering_models
def detect_by_leiden(src, trg, weight):
    g = igraph.Graph.TupleList(
        [[src[i], trg[i], weight[i]] for i in range(len(src))],
        weights=True,
        directed=False,
    )
    weights = [edge["weight"] for edge in g.es]
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights=weights
    )
    node_idx = np.array([g.vs[i]["name"] for i in range(len(g.vs))])
    memberships = np.zeros(net.shape[0])
    for i, p in enumerate(part):
        memberships[node_idx[p]] = i
    return memberships
