import torch
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.optim as optim
import networkx as nx
import numpy as np
import igraph



def initialize_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)





def igraph_to_pyg(igraph_graph):
    # Extract node features (identity matrix for illustration)
    num_nodes = igraph_graph.vcount()
    x = torch.eye(num_nodes)

    # Extract edge indices
    edge_index = torch.tensor(igraph_graph.get_edgelist(), dtype=torch.long).t().contiguous()

    # Extract edge weights as edge attributes
    #edge_attr = torch.tensor(igraph_graph.es['weight'], dtype=torch.float).view(-1, 1)

    # Create PyG Data object
    #data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = Data(x=x, edge_index=edge_index)

    return data





def igraph_to_pyg_weighted(igraph_graph):
    # Extract node features (identity matrix for illustration)
    num_nodes = igraph_graph.vcount()
    x = torch.eye(num_nodes)

    # Extract edge indices
    edge_index = torch.tensor(igraph_graph.get_edgelist(), dtype=torch.long).t().contiguous()

    # Extract edge weights as edge attributes
    edge_attr = torch.tensor(igraph_graph.es['weight'], dtype=torch.float).view(-1, 1)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #data = Data(x=x, edge_index=edge_index)

    return data








def nx_to_pyg(nx_graph):
    # Extract node features (identity matrix for illustration)
    num_nodes = len(nx_graph.nodes)
    x = torch.eye(num_nodes)

    # Extract edge indices
    edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()

    # Extract edge weights as edge attributes
    edge_attr = torch.tensor([edge[2]['weight'] for edge in nx_graph.edges(data=True)], dtype=torch.float).view(-1, 1)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data






# Define your GNN model
class LinkPredictionModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_edges):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(num_nodes, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # Adjust output size to match the number of edges

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

# Function to train the model
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            optimizer.zero_grad()

            output = model(data.x, data.edge_index)
            target = data.edge_attr.view(-1)
            #print('data.x.size', data.x.size())
            #print('data edge index size',data.edge_index.size())
            #print('output', output.size())
            #print('target',data.edge_attr.view(1,-1).size())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Function to predict weights for new graphs
def predict_weights(model, input_graph):
    input_data=igraph_to_pyg(input_graph)
    model.eval()
    with torch.no_grad():
        output = model(input_data.x, input_data.edge_index)
        predicted_weights = output.numpy()

        # Create a new PyG Data object with predicted weights
        predicted_data = Data(
            x=input_data.x,
            edge_index=input_data.edge_index,
            edge_attr=torch.tensor(predicted_weights, dtype=torch.float).view(-1, 1)
        )

    return pyg_to_igraph(predicted_data)




def pyg_to_igraph(pyg_data):

    # Extract node features (if available)
    if pyg_data.x is not None:
        node_features = pyg_data.x.numpy()
    else:
        node_features = None

    # Extract edge indices
    edge_indices = pyg_data.edge_index.numpy().T

    # Extract edge weights
    edge_weights = pyg_data.edge_attr.numpy().flatten()

    # Create an igraph Graph object
    igraph_graph = igraph.Graph(n=pyg_data.num_nodes, edges=edge_indices, directed=False)

    # Set node features (if available)
    if node_features is not None:
        igraph_graph.vs['features'] = node_features.tolist()

    # Set edge weights
    igraph_graph.es['weight'] = [abs(weight) for weight in edge_weights.tolist()]


    return igraph_graph




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