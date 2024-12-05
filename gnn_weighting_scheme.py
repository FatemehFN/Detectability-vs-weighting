import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import torch
import planted_partition_sadamori as PPS
import clustering_operations as CLO
import igraph
from scipy import sparse
import scipy.linalg


"""
def membership_matrix_to_vector(membership_matrix):
    num_nodes = membership_matrix.size(0)
    membership_vector = torch.zeros(num_nodes, dtype=torch.long)

    for i in range(num_nodes):
        # Check if all entries in the row are equal to the first entry
        if torch.all(membership_matrix[i] == membership_matrix[i, 0]):
            membership_vector[i] = membership_matrix[i, 0]

    return membership_vector.float()
"""


def nx_to_igraph(nx_graph):
    # Create an igraph graph
    igraph_graph = igraph.Graph(directed=nx_graph.is_directed())

    # Add vertices with attributes
    for node, attr in nx_graph.nodes(data=True):
        igraph_graph.add_vertex(name=str(node), **attr)

    # Add edges with attributes
    for edge in nx_graph.edges(data=True):
        source, target, attr = edge
        igraph_graph.add_edge(str(source), str(target), **attr)

    return igraph_graph


def predict_weights(g, weights):

    for src, dst, dict in g.edges(data=True):
        dict["weight"] = weights[src, dst].item()

    # print(g.edges(data=True))
    return nx_to_igraph(g)


def membership_vector_to_matrix(membership_vector):
    num_nodes = len(membership_vector)
    membership_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if membership_vector[i] == membership_vector[j]:
                membership_matrix[i, j] = 1
    # print('membership matrix',membership_matrix[100])
    return membership_matrix


class CommunityDetectionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CommunityDetectionGNN, self).__init__()
        self.conv1 = GraphSAGE(input_dim, hidden_dim, num_layers=1)
        self.conv2 = GraphSAGE(hidden_dim, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)

        # Compute dot similarity between every pair of node embeddings
        dot_similarity_matrix = torch.matmul(
            x, x.t()
        )  # x.t() transposes the node embeddings

        # Apply sigmoid activation function for binary classification
        out = torch.sigmoid(dot_similarity_matrix)
        # binary_out=(out >= 0.5).float()
        # return binary_out
        return out


# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-02 11:43:18
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-02 11:58:51


def LaplacianEigenMap(A, dim, return_basis_vector=False):
    """Laplacian EigenMap

    Example:
    >> import networkx as nx
    >> G = nx.karate_club_graph() # Get karate club net
    >> A = nx.adjacency_matrix(G) # To scipy sparse format
    >> emb = LaplacianEigenMap(A, dim=5) # Get the 5-dimensional embedding. Each column may have different norm.
    >> eig_emb = LaplacianEigenMap(A, dim=5, return_basis_vector=True) # Every column will be normalized to have a unit norm.

    :param A: Network
    :type A: scipy.sparse format
    :param dim: Dimension
    :type dim: int
    :return_basis_vector: Set True to obtain the eigenvectors of the Laplacian Matrix. Otherwise, return the projection of the (normalized) given network onto the space spanned by the Laplaian basis.
    :return: Embedding
    :rtype: numpy.ndarray of (num nodes, dim)
    """
    deg = np.array(A.sum(axis=1)).reshape(-1)
    Dinv = sparse.diags(1 / np.maximum(np.sqrt(deg), 1))
    L = Dinv @ A @ Dinv
    w, v = sparse.linalg.eigs(L, k=dim + 1)
    order = np.argsort(-w)
    v = np.real(v[:, order[1:]])
    w = np.real(w[order[1:]])

    if return_basis_vector:
        return v

    return L @ v


# Hyperparameters
num_nodes = 1000
N = num_nodes
num_epochs = 2000
hidden_dim = 64
input_dim = 64


# graph parameters
average_k = 10
q = 10
size_of_each_com = int(N / q)
ground_truth_membership_vector = np.concatenate(
    [np.full(size_of_each_com, i) for i in range(q)]
)
ground_truth_membership_matrix = membership_vector_to_matrix(
    ground_truth_membership_vector
)
ground_truth_memberships_tensor = torch.tensor(
    ground_truth_membership_matrix, dtype=torch.float
)
# print(ground_truth_memberships_tensor)
output_filename = str(average_k)
mus = np.arange(0, 0.75, 0.05)
# mus=[0.2]


# write results

f = open(
    f"/N/u/fsfatemi/Quartz/detectability_project/results/gnn_reweighting_scheme/gnn_reweighted_leiden_scores_{N}_nodes_{average_k}_k_{q}_q.txt",
    "w",
)


for mixing_rate in mus:
    score = []

    model = CommunityDetectionGNN(input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    model.train()
    for j in range(6):
        # print(j)
        adj_matrix_data = np.loadtxt(
            f"Networks/PP N {N} q {q}/k {average_k}/{average_k}_{mixing_rate}_{N}_{j}_adj_matrix.txt"
        )
        adj_matrix = adj_matrix_data.reshape(N, N)
        np.fill_diagonal(adj_matrix, 0)
        g = nx.Graph()
        num_vertices = adj_matrix.shape[0]
        for i in range(num_vertices):
            g.add_node(i)
        edge_list = [
            (i, k)
            for i in range(num_vertices)
            for k in range(i, num_vertices)
            if adj_matrix[i, k] != 0
        ]
        g.add_edges_from(edge_list)
        num_nodes = num_vertices
        num_features = num_vertices
        A = np.array(nx.to_numpy_array(g))
        node_features = LaplacianEigenMap(A, dim=input_dim)
        node_features = torch.from_numpy(node_features).to(torch.float32)
        # Convert the graph to PyTorch Geometric Data object
        edge_index = torch.tensor(
            np.array(nx.to_numpy_array(g).nonzero()), dtype=torch.long
        )
        data = Data(
            x=node_features, edge_index=edge_index, y=ground_truth_memberships_tensor
        )

        # Training loop
        for epoch in range(num_epochs):
            # print('epoch', epoch)
            optimizer.zero_grad()
            output = model(data)
            # print('output', output)
            target = ground_truth_memberships_tensor.requires_grad_(True)
            # print('target',target)
            # np.savetxt('tensor.txt', target.detach().numpy())
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            # print(output)
            # print('loss', loss.item())

        # model.eval()
        # print('output', output)

    # use the trained model
    model.eval()
    for j1 in range(6, 20):

        adj_matrix_data = np.loadtxt(
            f"Networks/PP N {N} q {q}/k {average_k}/{average_k}_{mixing_rate}_{N}_{j1}_adj_matrix.txt"
        )
        adj_matrix = adj_matrix_data.reshape(num_nodes, num_nodes)
        np.fill_diagonal(adj_matrix, 0)
        g = nx.Graph()
        num_vertices = adj_matrix.shape[0]
        for i in range(num_vertices):
            g.add_node(i)
        edge_list = [
            (i, z)
            for i in range(num_vertices)
            for z in range(i, num_vertices)
            if adj_matrix[i, z] != 0
        ]
        g.add_edges_from(edge_list)
        A = np.array(nx.to_numpy_array(g))
        node_features = LaplacianEigenMap(A, dim=input_dim)
        node_features = torch.from_numpy(node_features).to(torch.float32)
        edge_index = torch.tensor(
            np.array(nx.to_numpy_array(g).nonzero()), dtype=torch.long
        )
        data = Data(x=node_features, edge_index=edge_index)

        output = model(data)
        print("output", output)
        weighted_graph = predict_weights(g, output)
        part = CLO.leiden_max_modularity_part_weighted(weighted_graph, trials=100)
        s = PPS.calc_esim(
            ground_truth_membership_vector, part.membership, normalize=True
        )
        score.append(s)
    f.write(f"{mixing_rate} {sum(score) / len(score)}\n")
    print(mixing_rate, sum(score) / len(score))


f.close()
