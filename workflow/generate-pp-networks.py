# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-08 16:37:34
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 13:13:15
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
import graph_tool.all as gt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def generate_PPI_network(Cave, mixing_rate, N, q):
    """
    Generate PPI network.

    Cave: int. The average degree.
    mixing_rate: float. The mixing of communities, with range (0,1].
                The mixing_rate = 1 generates an Erdos-Renyi random graph, and mixing_rate~0 generates well-separated communities.
                It is defined as the ratio p_out / pave, where pout is the probability of inter-community edges, and
                pave is the average edge probability (the density of edges in the network).

    return: net, membership
        net: scipy.csr_matrix representation of the adjacency matrix of the generated network.
        membership: numpy array of membership IDs of the nodes in the network.

    """
    memberships = np.sort(np.arange(N) % q)
    #print(memberships)

    q = int(np.max(memberships) + 1)

    N = len(memberships)

    U = csr_matrix((np.ones(N), (np.arange(N), memberships)), shape=(N, q))
    #print(U)
    Cout = np.maximum(1, mixing_rate * Cave)
    Cin = q * Cave - (q - 1) * Cout
    pout = Cout / N
    pin = Cin / N

    Nk = np.array(U.sum(axis=0)).reshape(-1)

    P = np.ones((q, q)) * pout + np.eye(q) * (pin - pout)
    probs = np.diag(Nk) @ P @ np.diag(Nk)
    gt_params = {
        "b": memberships,
        "probs": probs,
        "micro_degs": False,
        "in_degs": np.ones_like(memberships) * Cave,
        "out_degs": np.ones_like(memberships) * Cave,
    }

    # Generate the network until the degree sequence
    # satisfied the thresholds
    while True:
        g = gt.generate_sbm(**gt_params)

        A = gt.adjacency(g).T

        A.data = np.ones_like(A.data)
        # check if the graph is connected
        if connected_components(A)[0] == 1:
            break
    return A, memberships











if "snakemake" in sys.modules:
    params = snakemake.params["parameters"]
    N = float(params["n"])
    k = float(params["k"])
    q = float(params["q"])
    mu = float(params["mu"])

    output_net_file = snakemake.output["output_file"]
    output_node_file = snakemake.output["output_node_file"]


else:
    input_file = "../data/"
    output_file = "../data/"

params = {
    "N": N,
    "k": k,
    "q":q,
    "mu": mu,
}

A,memberships=generate_PPI_network(Cave=k,mixing_rate=mu,N=N,q=q)
A.setdiag(0)

#save
A_sparse = sparse.csr_matrix(A)
sparse.save_npz(output_net_file, A_sparse)



pd.DataFrame(
    {"node_id": np.arange(len(memberships)), "membership": memberships}
).to_csv(output_node_file, index=False)
