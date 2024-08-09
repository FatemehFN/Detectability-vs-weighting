"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
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
        edge["weight"] = 1/betweenness
    
    return graph








def clustering(net, metric="euclidean"):
    src, trg, _ = sparse.find(net)


    g = igraph.Graph.TupleList(
        [[src[i], trg[i]] for i in range(len(src))],
        directed=False,
    )

    edge_betweenness = g.edge_betweenness(directed=False)
    
    # Assign edge betweenness as weights to the edges
    for edge, betweenness in zip(g.es, edge_betweenness):
        edge["weight"] = 1/betweenness


    weights = [edge["weight"] for edge in g.es]
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights=weights
    )
    node_idx = np.array([g.vs[i]["name"] for i in range(len(g.vs))])
    memberships = np.zeros(net.shape[0])
    for i, p in enumerate(part):
        memberships[node_idx[p]] = i

    return memberships


net = sparse.load_npz(net_file)




# Evaluate
group_ids = clustering(net, metric=metric)


# %%
# Save
#
np.savez(output_file, group_ids=group_ids)
