import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import gnn_operations as gnno
import torch
import planted_partition_sadamori as PPS
import clustering_operations as CLO






def predict_weights(g,weights):

    for src,dst,dict in g.edges(data=True):
        dict['weight']=weights[src,dst].item()

    #print(g.edges(data=True))
    return gnno.nx_to_igraph(g)

def membership_vector_to_matrix(membership_vector):
    num_nodes = len(membership_vector)
    membership_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if membership_vector[i] == membership_vector[j]:
                membership_matrix[i, j] = 1
    #print('membership matrix',membership_matrix[100])
    return membership_matrix

'''
def membership_matrix_to_vector(membership_matrix):
    num_nodes = membership_matrix.size(0)
    membership_vector = torch.zeros(num_nodes, dtype=torch.long)

    for i in range(num_nodes):
        # Check if all entries in the row are equal to the first entry
        if torch.all(membership_matrix[i] == membership_matrix[i, 0]):
            membership_vector[i] = membership_matrix[i, 0]

    return membership_vector.float()
'''




class CommunityDetectionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CommunityDetectionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim , 1000)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        # Compute dot similarity between every pair of node embeddings
        dot_similarity_matrix = torch.matmul(x, x.t()).requires_grad_(True) # x.t() transposes the node embeddings
        #print('dot similarity', dot_similarity_matrix)
        # Apply sigmoid activation function for binary classification
        out = torch.sigmoid(dot_similarity_matrix)
        #binary_out=(out >= 0.5).float()
        #return binary_out
        return out
        #return dot_similarity_matrix






# Hyperparameters
num_nodes = 1000
N=num_nodes

#graph parameters
average_k=10
q=20
size_of_each_com=int(N/q)
ground_truth_membership_vector = np.concatenate([np.full(size_of_each_com, i) for i in range(q)])
ground_truth_membership_matrix = membership_vector_to_matrix(ground_truth_membership_vector)
ground_truth_memberships_tensor = torch.tensor(ground_truth_membership_matrix, dtype=torch.float)
#print(ground_truth_memberships_tensor)
output_filename=str(average_k)
#mus=np.arange(0,0.75,0.05)
mus=[0.0]




#write results
f=open(f'results/trial 100 networks 20/community_level_gnn_reweighted_leiden_scores_1000_nodes_'+output_filename+'_'+str(q)+'.txt','w')




for mixing_rate in mus:
    score=[]
    #for j in range(20):
    for j in range(1):
        #print(j)
        adj_matrix_data = np.loadtxt('PP N 1000 q 20/k 10/'+output_filename+'_'+str(mixing_rate)+'_'+str(N)+'_'+
                                     str(j)+'_'+'adj_matrix.txt')


        # Reshape the 1D array into a N*N matrix
        adj_matrix = adj_matrix_data.reshape(N, N)
        np.fill_diagonal(adj_matrix, 0)
        g=nx.Graph()
        num_vertices = adj_matrix.shape[0]

        # Add vertices to the graph
        for i in range(num_vertices):
            g.add_node(i)


        edge_list = [(i, j) for i in range(num_vertices) for j in range(i , num_vertices) if adj_matrix[i, j] != 0]

        g.add_edges_from(edge_list)

        # Define node features (identity features)
        num_nodes = num_vertices
        num_features = num_vertices
        node_features = torch.eye(num_nodes, num_features)


        # Convert the graph to PyTorch Geometric Data object
        edge_index = torch.tensor(np.array(nx.to_numpy_array(g).nonzero()), dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index, y=ground_truth_memberships_tensor)

        num_epochs = 20
        input_dim = 100
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
            print('output', output)
            target = ground_truth_memberships_tensor.requires_grad_(True)
            #print('target',target)
            #np.savetxt('tensor.txt', target.detach().numpy())
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            #print(output)
            print('loss', loss.item())



        weighted_graph=predict_weights(g,output)
        part=CLO.leiden_max_modularity_part_weighted(weighted_graph, trials=100)

        s = PPS.calc_esim(ground_truth_membership_vector, part.membership, normalize=True)
        score.append(s)
    f.write(str(mixing_rate) + ' ' + str((sum(score) / len(score))) + '\n')
    print(mixing_rate,(sum(score) / len(score)))
f.close()







'''
# Evaluation
model.eval()
with torch.no_grad():
    output = model(data)
    predictions = (output > 0.5).long()  # Convert probabilities to binary predictions
    accuracy = (predictions == data.y).sum().item() / len(data.y)
    print("Accuracy:", accuracy)
'''