"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys
from sklearn import cluster
import igraph
import numpy as np
import pandas as pd
from scipy import sparse, stats
from clustering import clustering_models

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    reweighting_method = params["reweighting"]
    clustering_method = params["clustering"]
    # model_name = params["model_name"]

else:
    net_file = "../data/derived/community-detection-datasets/mpm/networks/net_n~5000_q~50_cave~50_mu~0.25_sample~8.npz"
    com_file = "../data/derived/community-detection-datasets/mpm/networks/node_n~5000_q~50_cave~50_mu~0.25_sample~8.npz"
    emb_file = "../data/derived/community-detection-datasets/mpm/embedding/n~5000_q~50_cave~50_mu~0.25_sample~8_model~GAT_dim~128.npz"
    model_name = "GAT"
    output_file = "unko"
    metric = "cosine"


def bc_reweight(graph):
    """
    Reweights the edges of the graph based on edge betweenness centrality.

    Parameters:
        graph (igraph.Graph): The input graph.

    Returns:
        igraph.Graph: The weighted graph with edge weights based on betweenness centrality.
    """
    # Calculate edge betweenness centrality
    edge_betweenness = graph.edge_betweenness(directed=False)

    # Assign edge betweenness as weights to the edges
    for edge, betweenness in zip(graph.es, edge_betweenness):
        edge["weight"] = 1 / betweenness

    return graph


# Fatameh, if you want to add a new reweighting method, you can do so by:
# 1) Adding a new function below with any name you like
# 2) Add "elif" clause to the reweighting_method variable, with condition that reweighting_method == <your_new_method_name>. This "<your_new_method_name>" is any string consisting of alphabets (no numbers or special characters).
# 3) Then, add the <your_new_method_name> to the list in Snakemake in a dictionary named `params_topology_reweighting`


def scipy2igraph(net):
    src, trg, weight = sparse.find(net)
    return igraph.Graph.TupleList(
        [[src[i], trg[i], weight[i]] for i in range(len(src))],
        weights=True,
        directed=False,
    )


net = sparse.load_npz(net_file)

g = scipy2igraph(net)

if reweighting_method == "bc":
    g = bc_reweight(g)
else:
    raise ValueError(f"Unknown reweighting method: {reweighting_method}")

group_ids = clustering_models[clustering_method](g)

np.savez(output_file, group_ids=group_ids)
