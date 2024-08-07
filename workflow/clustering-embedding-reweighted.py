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
    com_detect_method = params["clustering"]
    # model_name = params["model_name"]

else:
    net_file = "../data/derived/community-detection-datasets/mpm/networks/net_n~5000_q~50_cave~50_mu~0.25_sample~8.npz"
    com_file = "../data/derived/community-detection-datasets/mpm/networks/node_n~5000_q~50_cave~50_mu~0.25_sample~8.npz"
    emb_file = "../data/derived/community-detection-datasets/mpm/embedding/n~5000_q~50_cave~50_mu~0.25_sample~8_model~GAT_dim~128.npz"
    model_name = "GAT"
    output_file = "unko"
    com_detect_method = "infomap"
    metric = "cosine"


from sklearn import cluster

import igraph
import leidenalg
import infomap


def clustering(net, emb, group_ids, com_detect_method="leiden", metric="euclidean"):
    src, trg, _ = sparse.find(net)

    if metric == "dotsim":
        weight = np.sum(emb[src, :] * emb[trg, :], axis=1).reshape(-1)
        weight = np.maximum(weight, 1e-10)
    elif metric == "cosine":
        nemb = emb / np.linalg.norm(emb, axis=1).reshape(-1, 1)
        weight = nemb[src, :] * nemb[trg, :]
        weight = np.sum(weight, axis=1).reshape(-1)
        weight = np.maximum(weight, 1e-10)
    elif metric == "sigmoid":
        weight = np.sum(emb[src, :] * emb[trg, :], axis=1).reshape(-1)
        weight = 1.0 / (1 + np.exp(-weight))
    else:
        weight = np.linalg.norm(emb[src, :] - emb[trg, :], axis=1)
    n_nodes = net.shape[0]
    if com_detect_method == "leiden":
        memberships = detect_by_leiden(src, trg, weight)
    elif com_detect_method == "infomap":
        memberships = detect_by_infomap(src, trg, weight, n_nodes)
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
group_ids = clustering(
    net, emb, memberships, com_detect_method=com_detect_method, metric=metric
)
# %%
group_ids_ = np.zeros(n_nodes) * np.nan
group_ids_[keep] = group_ids

# %%
# Save
#
np.savez(output_file, group_ids=group_ids_)
