import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import Data
import numpy as np

def membership_vector_to_matrix(membership_vector):
    num_nodes = len(membership_vector)
    membership_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if membership_vector[i] == membership_vector[j]:
                membership_matrix[i, j] = 1

    return membership_matrix


def membership_matrix_to_vector(membership_matrix):
    num_nodes = membership_matrix.size(0)
    membership_vector = torch.zeros(num_nodes, dtype=torch.long)

    for i in range(num_nodes):
        # Check if all entries in the row are equal to the first entry
        if torch.all(membership_matrix[i] == membership_matrix[i, 0]):
            membership_vector[i] = membership_matrix[i, 0]

    return membership_vector.float()





class CommunityDetectionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CommunityDetectionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim , input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Compute dot similarity between every pair of node embeddings
        dot_similarity_matrix = torch.matmul(x, x.t()).requires_grad_(True) # x.t() transposes the node embeddings

        # Apply sigmoid activation function for binary classification
        out = torch.sigmoid(dot_similarity_matrix)
        #binary_out=(out >= 0.5).float()
        #return binary_out
        return out

# Example usage
G = nx.Graph()
G.add_nodes_from(range(5))
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

# Define node features (identity features)
num_nodes = len(G.nodes)
num_features = 5
node_features = torch.eye(num_nodes, num_features)

# Define community memberships
ground_truth_membership_vector=[0, 0, 0, 1, 1]
ground_truth_membership_matrix=membership_vector_to_matrix(ground_truth_membership_vector)
ground_truth_memberships_tensor = torch.tensor(ground_truth_membership_matrix, dtype=torch.float)

# Convert the graph to PyTorch Geometric Data object
edge_index = torch.tensor(np.array(nx.to_numpy_array(G).nonzero()), dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index, y=ground_truth_memberships_tensor)




num_epochs=20
input_dim = data.num_node_features
hidden_dim = 64
model = CommunityDetectionGNN(input_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    print('epoch', epoch)
    optimizer.zero_grad()
    output = model(data)
    target = ground_truth_memberships_tensor.requires_grad_(True)
    print('target',target)
    print('output',output)
    #print(target)
    loss = criterion(output, target)  # Assuming data.y contains the ground truth labels
    loss.backward()

    optimizer.step()
    print('loss', loss.item())
print(output)
# Evaluation
model.eval()
with torch.no_grad():
    output = model(data)
    predictions = (output > 0.5).long()  # Convert probabilities to binary predictions
    accuracy = (predictions == data.y).sum().item() / len(data.y)
    print("Accuracy:", accuracy)
