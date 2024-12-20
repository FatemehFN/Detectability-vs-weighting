import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import os
from gnn_tools.models import embedding_models

configfile: "workflow/config.yaml"
include: "./workflow/workflow_utils.smk"  # not able to merge this with snakemake_utils.py due to some path breakage issues


# ====================
# Root folder path setting
# ====================

# network file
DATA_DIR = config["data_dir"]  # set test_data for testing

DERIVED_DIR = j(DATA_DIR, "derived")
EMB_DIR = j(DERIVED_DIR, "embedding")
FIG_DIR =j("figs")

# Multi partition model
N_SAMPLES = 10


#
# Network embedding
#
MODEL_LIST = ["GAT"]
#MODEL_LIST = ["GraphSAGE", "GIN", "GAT", "GCN", "dcGraphSAGE", "dcGIN", "dcGAT", "dcGCN"]
params_emb = {"model": MODEL_LIST, "dim": [128]}
paramspace_emb = to_paramspace(params_emb)
#
# Community detection
#

# Multi partition model
params_mpm = {
    "n": [5000],  # Network size
    "q": [50],  # Number of communities
    "cave": [10, 50],  # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": list(range(N_SAMPLES)),  # Number of samples
}

params_lfr = { # LFR
    "n": [3000],  # Network size
    "k": [50],  # Average degree
    "tau": [3],  # degree exponent
    "tau2": [3],  # community size exponent
    "minc": [100],  # min community size
    "maxk": [1000], # maximum degree,
    "maxc": [1000], # maximum community size
    "mu": [0.1],
    "sample": [1],  # Number of samples
}

params_clustering_embedding_reweighting = {
    "metric": ["cosine", "dotsim", "sigmoid"],
    "clustering": ["leiden"],
}

params_clustering_topology_reweighting = {
    "metric": ["bc"],
    "clustering": ["leiden", "infomap"],
}

params_fig_lfr = {
    "n": params_lfr["n"],
    "k": params_lfr["k"],
    "tau": params_lfr["tau"],
    "dim": params_emb["dim"],
    "minc": params_lfr["minc"],
    "maxk": params_lfr["maxk"],
    "maxc": params_lfr["maxc"],
    "metric": params_clustering_embedding_reweighting["metric"],
    "clustering": params_clustering_embedding_reweighting["clustering"],
}
params_fig_mpm = {
    "n": params_mpm["n"],
    "q": params_mpm["q"],
    "dim": params_emb["dim"],
    "cave": params_mpm["cave"],
    "metric": params_clustering_embedding_reweighting["metric"],
    "clustering": params_clustering_embedding_reweighting["clustering"],
}

# ======================================
# Community Detection Benchmark Datasets
# ======================================

CMD_DATASET_DIR = j(DERIVED_DIR, "community-detection-datasets")

# LFR benchmark
LFR_DIR = j(CMD_DATASET_DIR, "lfr")

LFR_NET_DIR = j(LFR_DIR, "networks")
LFR_EMB_DIR = j(LFR_DIR, "embedding")
LFR_CLUST_DIR = j(LFR_DIR, "clustering")
LFR_EVAL_DIR = j(LFR_DIR, "evaluations")

paramspace_lfr = to_paramspace(params_lfr)
LFR_NET_FILE = j(LFR_NET_DIR, f"net_{paramspace_lfr.wildcard_pattern}.npz")
LFR_NODE_FILE = j(LFR_NET_DIR, f"node_{paramspace_lfr.wildcard_pattern}.npz")
LFR_NET_TRAIN_FILE = j(LFR_NET_DIR, f"train_net_{paramspace_lfr.wildcard_pattern}.npz")
LFR_NODE_TRAIN_FILE = j(LFR_NET_DIR, f"train_node_{paramspace_lfr.wildcard_pattern}.npz")

paramspace_lfr_emb = to_paramspace([params_lfr, params_emb])
LFR_EMB_FILE = j(LFR_EMB_DIR, f"{paramspace_lfr_emb.wildcard_pattern}.npz")

paramspace_lfr_com_detect_emb = to_paramspace([params_lfr, params_emb, params_clustering_embedding_reweighting])
paramspace_lfr_com_detect_topo = to_paramspace([params_lfr, params_clustering_topology_reweighting])

LFR_COM_DETECT_EMB_FILE = j(
    LFR_CLUST_DIR, f"clus_{paramspace_lfr_com_detect_emb.wildcard_pattern}.npz"
)

LFR_COM_DETECT_TOPO_FILE = j(
    LFR_CLUST_DIR, f"clus_topo_reweighted_{paramspace_lfr_com_detect_topo.wildcard_pattern}.npz"
)


LFR_EVAL_EMB_FILE = j(LFR_EVAL_DIR, f"score_clus_{paramspace_lfr_com_detect_emb.wildcard_pattern}.npz")
LFR_EVAL_TOPO_FILE = j(LFR_EVAL_DIR, f"score_clus_topo_reweighted_{paramspace_lfr_com_detect_topo.wildcard_pattern}.npz")

# Figure
FIG_LFR_PERF_CURVE = j(FIG_DIR, "lfr_perf_curve_n~{n}_k~{k}_tau~{tau}_dim~{dim}_minc~{minc}_maxk~{maxk}_maxc~{maxc}_metric~{metric}.pdf")
FIG_LFR_AUCESIM = j(FIG_DIR, "lfr_aucesim_n~{n}_k~{k}_tau~{tau}_dim~{dim}_minc~{minc}_maxk~{maxk}_maxc~{maxc}_metric~{metric}.pdf")

FIG_LFR_PERF_CURVE_NMI = j(FIG_DIR, "lfr_perf_curve_metric~nmi_n~{n}_k~{k}_tau~{tau}_dim~{dim}_minc~{minc}_maxk~{maxk}_maxc~{maxc}_metric~{metric}.pdf")
FIG_LFR_AUCNMI = j(FIG_DIR, "lfr_aucesim_metric~nmi_n~{n}_k~{k}_tau~{tau}_dim~{dim}_minc~{minc}_maxk~{maxk}_maxc~{maxc}_metric~{metric}.pdf")

FIG_MPM_PERF_CURVE = j(FIG_DIR, "mpm_perf_curve_n~{n}_q~{q}_cave~{cave}_dim~{dim}_metric~{metric}.pdf")
FIG_MPM_AUCESIM = j(FIG_DIR, "mpm_aucesim_n~{n}_q~{q}_cave~{cave}_dim~{dim}_metric~{metric}.pdf")

FIG_MPM_PERF_CURVE_NMI = j(FIG_DIR, "mpm_perf_curve_metric~nmi_n~{n}_q~{q}_cave~{cave}_dim~{dim}_metric~{metric}.pdf")
FIG_MPM_AUCNMI = j(FIG_DIR, "mpm_aucesim_metric~nmi_n~{n}_q~{q}_cave~{cave}_dim~{dim}_metric~{metric}.pdf")

# Multi partition model
MPM_DIR = j(CMD_DATASET_DIR, "mpm")

MPM_NET_DIR = j(MPM_DIR, "networks")
MPM_EMB_DIR = j(MPM_DIR, "embedding")
MPM_CLUST_DIR = j(MPM_DIR, "clustering")
MPM_EVAL_DIR = j(MPM_DIR, "evaluations")

paramspace_mpm = to_paramspace(params_mpm)
MPM_NET_FILE = j(MPM_NET_DIR, f"net_{paramspace_mpm.wildcard_pattern}.npz")
MPM_NODE_FILE = j(MPM_NET_DIR, f"node_{paramspace_mpm.wildcard_pattern}.npz")
MPM_NET_TRAIN_FILE = j(MPM_NET_DIR, f"train_net_{paramspace_mpm.wildcard_pattern}.npz")
MPM_NODE_TRAIN_FILE = j(MPM_NET_DIR, f"train_node_{paramspace_mpm.wildcard_pattern}.npz")

paramspace_mpm_emb = to_paramspace([params_mpm, params_emb])
MPM_EMB_FILE = j(MPM_EMB_DIR, f"{paramspace_mpm_emb.wildcard_pattern}.npz")

paramspace_mpm_com_detect_emb = to_paramspace([params_mpm, params_emb, params_clustering_embedding_reweighting])
paramspace_mpm_com_detect_topo = to_paramspace([params_mpm, params_clustering_topology_reweighting])

MPM_COM_DETECT_EMB_FILE = j(
    MPM_CLUST_DIR, f"clus_{paramspace_mpm_com_detect_emb.wildcard_pattern}.npz"
)
MPM_COM_DETECT_TOPO_FILE = j(
    MPM_CLUST_DIR, f"clus_topo_reweighted_{paramspace_mpm_com_detect_topo.wildcard_pattern}.npz"
)
MPM_EVAL_EMB_FILE = j(MPM_EVAL_DIR, f"score_clus_emb_{paramspace_mpm_com_detect_emb.wildcard_pattern}.npz")
MPM_EVAL_TOPO_FILE = j(MPM_EVAL_DIR, f"score_clus_topo_reweighted_{paramspace_mpm_com_detect_topo.wildcard_pattern}.npz")

# ======================================
# LFR benchmark
# ======================================

rule all_mpm:
    input:
        expand(MPM_EVAL_EMB_FILE, **params_mpm, **params_emb, **params_clustering_embedding_reweighting)+expand(MPM_EVAL_TOPO_FILE, **params_mpm, **params_clustering_topology_reweighting),
        j(MPM_EVAL_DIR, "all_scores.csv"),
        expand(FIG_MPM_PERF_CURVE, **params_fig_mpm),
        expand(FIG_MPM_AUCESIM, **params_fig_mpm),

rule all_lfr:
    input:
        expand(LFR_EVAL_EMB_FILE, **params_lfr, **params_emb, **params_clustering_embedding_reweighting)+expand(LFR_EVAL_TOPO_FILE, **params_lfr, **params_clustering_topology_reweighting),
        j(LFR_EVAL_DIR, "all_scores.csv"),

rule figs_lfr:
    input:
        expand(FIG_LFR_PERF_CURVE, **params_fig_lfr),
        expand(FIG_LFR_AUCESIM, **params_fig_lfr),
        expand(FIG_LFR_PERF_CURVE_NMI, **params_fig_lfr),
        expand(FIG_LFR_AUCNMI, **params_fig_lfr),


rule generate_lfr_net_train:
    params:
        parameters=paramspace_lfr.instance,
    output:
        output_file=LFR_NET_TRAIN_FILE,
        output_node_file=LFR_NODE_TRAIN_FILE,
    wildcard_constraints:
        data="lfr"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/generate-lfr-networks.py"

rule generate_lfr_net:
    params:
        parameters=paramspace_lfr.instance,
    output:
        output_file=LFR_NET_FILE,
        output_node_file=LFR_NODE_FILE,
    wildcard_constraints:
        data="lfr"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/generate-lfr-networks.py"

rule embedding_lfr:
    input:
        train_net_file=LFR_NET_TRAIN_FILE,
        train_com_file=LFR_NODE_TRAIN_FILE,
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EMB_FILE,
    params:
        parameters=paramspace_emb.instance,
    script:
        "workflow/embedding-supervised-gnn.py"

rule kmeans_clustering_lfr:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=paramspace_lfr_com_detect_emb.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/kmeans-clustering.py"

rule clustering_lfr_embedding_reweighted:
    input:
        net_file = LFR_NET_FILE,
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=paramspace_lfr_com_detect_emb.instance,
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/clustering-embedding-reweighted.py"

rule clustering_lfr_topology_reweighted:
    input:
        net_file = LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_TOPO_FILE,
    params:
        parameters=paramspace_lfr_com_detect_topo.instance,
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/clustering-topology-reweighted.py"

rule evaluate_communities_lfr:
    input:
        detected_group_file=LFR_COM_DETECT_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_EMB_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "workflow/eval-com-detect-score.py"

rule evaluate_communities_lfr_topo_reweighted:
    input:
        detected_group_file=LFR_COM_DETECT_TOPO_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_TOPO_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "workflow/eval-com-detect-score.py"

rule concatenate_lfr_result:
    input:
        input_files = expand(LFR_EVAL_EMB_FILE, **params_lfr, **params_emb, **params_clustering_embedding_reweighting)+expand(LFR_EVAL_TOPO_FILE, **params_lfr, **params_clustering_topology_reweighting),
    output:
        output_file = j(LFR_EVAL_DIR, "all_scores.csv"),
    params:
        to_int = ["n", "k", "tau2", "minc", "dim", "sample"],
        to_float = ["mu", "tau"],
    script:
        "workflow/concatenate-com-detect-results.py"

rule plot_lfr_result:
    input:
        input_file = j(LFR_EVAL_DIR, "all_scores.csv"),
    output:
        output_file_performance = FIG_LFR_PERF_CURVE,
        output_file_aucesim = FIG_LFR_AUCESIM,
    params:
        model = ["GIN", "GCN", "GAT", "GraphSAGE", "dcGIN", "dcGCN", "dcGAT", "dcGraphSAGE"],
        clustering = "kmeans",
        metric = "cosine",
        score_type = "esim",
        tau = lambda wildcards: float(wildcards.tau),
        k = lambda wildcards: int(wildcards.k),
        n = lambda wildcards: int(wildcards.n),
        dim = lambda wildcards: int(wildcards.dim),
        minc = lambda wildcards: int(wildcards.minc),
        maxk = lambda wildcards: int(wildcards.maxk),
        maxc = lambda wildcards: int(wildcards.maxc),
    script:
        "workflow/plot_lfr_scores.py"

rule plot_lfr_result_nmi:
    input:
        input_file = j(LFR_EVAL_DIR, "all_scores.csv"),
    output:
        output_file_performance = FIG_LFR_PERF_CURVE_NMI,
        output_file_aucesim = FIG_LFR_AUCNMI,
    params:
        model = ["GIN", "GCN", "GAT", "GraphSAGE", "dcGIN", "dcGCN", "dcGAT", "dcGraphSAGE"],
        clustering = "kmeans",
        score_type = "nmi",
        tau = lambda wildcards: float(wildcards.tau),
        k = lambda wildcards: int(wildcards.k),
        n = lambda wildcards: int(wildcards.n),
        dim = lambda wildcards: int(wildcards.dim),
        minc = lambda wildcards: int(wildcards.minc),
        maxk = lambda wildcards: int(wildcards.maxk),
        maxc = lambda wildcards: int(wildcards.maxc),
        metric = lambda wildcards: wildcards.metric,
    script:
        "workflow/plot_lfr_scores.py"

# ======================================
# MPM benchmark
# ======================================

rule generate_mpm_net_train:
    params:
        parameters=paramspace_mpm.instance,
    output:
        output_file=MPM_NET_TRAIN_FILE,
        output_node_file=MPM_NODE_TRAIN_FILE,
    wildcard_constraints:
        data="mpm"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/generate-mpm-networks.py"

rule generate_mpm_net:
    params:
        parameters=paramspace_mpm.instance,
    output:
        output_file=MPM_NET_FILE,
        output_node_file=MPM_NODE_FILE,
    wildcard_constraints:
        data="mpm"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/generate-mpm-networks.py"

rule embedding_mpm:
    input:
        train_net_file=MPM_NET_TRAIN_FILE,
        train_com_file=MPM_NODE_TRAIN_FILE,
        net_file=MPM_NET_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_EMB_FILE,
    params:
        parameters=paramspace_emb.instance,
    script:
        "workflow/embedding-supervised-gnn.py"

rule kmeans_clustering_mpm:
    input:
        emb_file=MPM_EMB_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_COM_DETECT_EMB_FILE,
    params:
        parameters=paramspace_mpm_com_detect_emb.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/kmeans-clustering.py"

rule clustering_mpm_embedding_reweighted:
    input:
        net_file = MPM_NET_FILE,
        emb_file=MPM_EMB_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_COM_DETECT_EMB_FILE,
    params:
        parameters=paramspace_mpm_com_detect_emb.instance,
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/clustering-embedding-reweighted.py"

rule clustering_mpm_topology_reweighted:
    input:
        net_file = MPM_NET_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_COM_DETECT_TOPO_FILE,
    params:
        parameters=paramspace_mpm_com_detect_topo.instance,
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/clustering-topology-reweighted.py"

rule evaluate_communities_mpm:
    input:
        detected_group_file=MPM_COM_DETECT_EMB_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_EVAL_EMB_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "workflow/eval-com-detect-score.py"

rule evaluate_communities_mpm_topo_reweighted:
    input:
        detected_group_file=MPM_COM_DETECT_TOPO_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_EVAL_TOPO_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "workflow/eval-com-detect-score.py"

rule concatenate_mpm_result:
    input:
        input_files = expand(MPM_EVAL_EMB_FILE, **params_mpm, **params_emb, **params_clustering_embedding_reweighting) + expand(MPM_EVAL_TOPO_FILE, **params_mpm, **params_clustering_topology_reweighting),
    output:
        output_file = j(MPM_EVAL_DIR, "all_scores.csv"),
    params:
        to_int = ["n", "q", "dim", "sample", "cave"],
        to_float = ["mu"]
    script:
        "workflow/concatenate-com-detect-results.py"


rule plot_mpm_result:
    input:
        input_file = j(MPM_EVAL_DIR, "all_scores.csv"),
    output:
        output_file_performance = FIG_MPM_PERF_CURVE,
        output_file_aucesim = FIG_MPM_AUCESIM,
    params:
        model = ["GAT"],
        clustering = "leiden",
        score_type = "esim",
        q = lambda wildcards: int(wildcards.q),
        n = lambda wildcards: int(wildcards.n),
        dim = lambda wildcards: int(wildcards.dim),
        cave = lambda wildcards: int(wildcards.cave),
        metric = lambda wildcards: wildcards.metric,
    script:
        "workflow/plot_mpm_scores.py"

