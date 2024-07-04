# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-10 08:46:51
# %%
import sys
import numpy as np
from scipy import sparse
import pandas as pd
from gnn_tools.models import *
import gnn_tools

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    train_netfile = snakemake.input["train_net_file"]
    train_com_file = snakemake.input["train_com_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    train_netfile = "../data/derived/community-detection-datasets/mpm/networks/train_net_n~500_q~50_cave~10_mu~0.70_sample~9.npz"
    train_com_file = "../data/derived/community-detection-datasets/mpm/networks/train_node_n~500_q~50_cave~10_mu~0.70_sample~9.npz"
    netfile = "../data/derived/community-detection-datasets/mpm/networks/net_n~500_q~50_cave~10_mu~0.70_sample~9.npz"
    comfile = "data/derived/community-detection-datasets/mpm/networks/node_n~500_q~50_cave~10_mu~0.70_sample~9.npz"
    embfile = "tmp.npz"
    params = {"model": "GraphSAGE", "dim": 128}

dim = int(params["dim"])
model_name = params["model"]


def load_net(netfile):
    net = sparse.load_npz(netfile)
    net = net + net.T
    net.data = net.data * 0.0 + 1.0
    return net


net = load_net(netfile)
train_net = load_net(train_netfile)


memberships = pd.read_csv(train_com_file)["membership"].values.astype(int)

#
# Embedding
#
net = sparse.csr_matrix(net)

# Embedding
dim = np.minimum(dim, net.shape[0] - 5)

# Training
emb, model, device = embedding_models[model_name](
    train_net, dim=dim, return_model=True, memberships=memberships
)
# %%
# Embedding
emb = gnn_tools.train.generate_embedding(model, net, device=device)

#
# Save
#
np.savez_compressed(
    embfile,
    emb=emb,
    dim=dim,
    model_name=model_name,
)
