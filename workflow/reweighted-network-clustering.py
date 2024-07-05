"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    # model_name = params["model_name"]

else:
    net_file = "../data/derived/community-detection-datasets/mpm/networks/net_n~5000_q~50_cave~50_mu~0.25_sample~8.npz"
    com_file = "../data/derived/community-detection-datasets/mpm/networks/node_n~5000_q~50_cave~50_mu~0.25_sample~8.npz"
    emb_file = "../data/derived/community-detection-datasets/mpm/embedding/n~5000_q~50_cave~50_mu~0.25_sample~8_model~GAT_dim~128.npz"
    model_name = "GAT"
    output_file = "unko"
    metric = "cosine"


from sklearn import cluster

import igraph
import leidenalg


def clustering(net, emb, group_ids, metric="euclidean"):
    src, trg, _ = sparse.find(net)

    if metric == "cosine":
        nemb = emb / np.linalg.norm(emb, axis=1).reshape(-1, 1)
        weight = nemb[src, :] * nemb[trg, :]
        weight = np.sum(weight, axis=1).reshape(-1)
    else:
        weight = np.linalg.norm(emb[src, :] - emb[trg, :], axis=1)

    W = sparse.csr_matrix((weight, (src, trg)), shape=(emb.shape[0], emb.shape[0]))

    g = igraph.Graph.TupleList([[src[i], trg[i], weight[i]] for i in range(len(src))])
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    memberships = np.zeros(net.shape[0])
    for i, p in enumerate(part):
        memberships[p] = i

    node_idx = np.argsort([g.vs[i]["name"] for i in range(len(g.vs))])
    memberships = memberships[node_idx]

    return memberships


# Load emebdding
net = sparse.load_npz(net_file)
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Remove nan embedding
remove = np.isnan(np.array(np.sum(emb, axis=1)).reshape(-1))
keep = np.where(~remove)[0]
n_nodes = emb.shape[0]
emb = emb[keep, :]
memberships = memberships[keep]

# Evaluate
group_ids = clustering(net, emb, memberships, metric=metric)

group_ids_ = np.zeros(n_nodes) * np.nan
group_ids_[keep] = group_ids

# %%
# Save
#
np.savez(output_file, group_ids=group_ids_)
