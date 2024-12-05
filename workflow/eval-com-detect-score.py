# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-13 16:13:54
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-13 16:15:00
"""Evaluate the detected communities using the element-centric similarity."""

# %%
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics.cluster import normalized_mutual_info_score

if "snakemake" in sys.modules:
    detected_group_file = snakemake.input["detected_group_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
else:
    com_file = "../data/derived/community-detection-datasets/mpm/networks/node_n~5000_q~50_cave~50_mu~0.10_sample~5.npz"
    detected_group_file = "../data/derived/community-detection-datasets/mpm/clustering/clus_n~5000_q~50_cave~50_mu~0.10_sample~5_model~GAT_dim~128_metric~cosine_clustering~leiden.npz"
    output_file = "unko"
    scoreType = "esim"


#
# Load
#
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
groups = np.load(detected_group_file)


#
# Evaluation
#
def calc_esim(y, ypred):
    """Element centric similarity."""
    ylab, y = np.unique(y, return_inverse=True)
    ypredlab, ypred = np.unique(ypred, return_inverse=True)

    Ka, Kb = len(ylab), len(ypredlab)

    K = np.maximum(Ka, Kb)
    N = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
    )

    nA = np.array(UA.sum(axis=0)).reshape(-1)
    nB = np.array(UB.sum(axis=0)).reshape(-1)

    nAB = (UA.T @ UB).toarray()
    nAB_rand = np.outer(nA, nB) / N

    # Calc element-centric similarity
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    S = np.sum(np.multiply(Q, (nAB**2))) / N

    # Calc the expected element-centric similarity for random partitions
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    Srand = np.sum(np.multiply(Q, (nAB_rand**2))) / N
    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected


def get_params(keys, sep="~"):
    params = keys.split("_")
    retval = {}
    for p in params:
        if sep not in p:
            continue
        kv = p.split(sep)
        retval[kv[0]] = kv[1]
    return retval


results = []
for keys, group_ids in groups.items():
    s = ~np.isnan(group_ids)
    memberships_, group_ids_ = memberships[s], group_ids[s]
    for scoreType in ["nmi", "esim"]:
        if scoreType == "nmi":
            score = normalized_mutual_info_score(memberships_, group_ids_)
        elif scoreType == "esim":
            score = calc_esim(memberships_, group_ids_)
        else:
            raise ValueError("Unknown score type: {}".format(scoreType))
        params = get_params(keys)
        results.append({"score": score, "score_type": scoreType, **params})
#
# Save
#
res_table = pd.DataFrame(results)
res_table

# %%

res_table.to_csv(output_file, index=False)

# %%
