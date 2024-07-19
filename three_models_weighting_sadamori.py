
from scipy import sparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from sklearn.metrics import roc_auc_score
import clustering_operations as CLO
import igraph
import planted_partition_sadamori as PPS
import networkx as nx
import torch
import gnn_tools



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



def predict_weights(g,weights):

    for src,dst,dict in g.edges(data=True):
        dict['weight']=weights[src,dst].item()

    #print(g.edges(data=True))
    return nx_to_igraph(g)



#print(torch.cuda.is_available())




# To make sure that the version is correct. If failed, please re-install the package.
assert gnn_tools.__version__ == "0.1"

num_nodes = 10000
N=num_nodes
average_k=10
#qs=[2,10,20]

qs=[2,10,20]

mus=np.arange(0,1.05,0.05)







for q in qs:
    f=open(f'/N/u/fsfatemi/Quartz/detectability_project/results/three_gnn_reweighting_scheme/three_gnn_reweighted_LA_scores_{N}_nodes_{average_k}_k_{q}_q.txt','w')
    size_of_each_com=int(N/q)
    labels = np.concatenate([np.full(size_of_each_com, i) for i in range(q)])

    for mixing_rate in mus:
        score=[]

        for j in range(20):



            # Load data
            A = sparse.load_npz(f'Networks/PP N {N} q {q}/k {average_k}/{average_k}_{mixing_rate}_{N}_{j}_adj_matrix.npz')
            adj_matrix_data=A.toarray()
            # Reshape the 1D array into a N*N matrix
            adj_matrix = adj_matrix_data.reshape(N, N)

            #get the nx graph from adjacency matrix
            g=nx.Graph()
            num_vertices = adj_matrix.shape[0]
            for i in range(num_vertices):
                g.add_node(i)
            edge_list = [(i, k) for i in range(num_vertices) for k in range(i , num_vertices) if A[i, k] != 0]
            g.add_edges_from(edge_list)


            #go through models and find the best one for weighting
            model_names = ["GCN", "GAT", "GraphSAGE"]
            best_score_roc_auc=0
            for model_name in model_names:
                emb = gnn_tools.embedding_models[model_name](A, dim=128, memberships=labels)

                S = emb @ emb.T
                U = sparse.csr_matrix(
                    (np.ones_like(labels), (np.arange(len(labels)), labels)),
                    shape=(len(labels), len(set(labels))),
                )
                Sy = (U @ U.T).toarray()

                s_model = roc_auc_score(Sy.reshape(-1), S.reshape(-1))
                if s_model>best_score_roc_auc:
                    emb_to_use=S
                    best_score_roc_auc=s_model


            #weight the graph with the embedding found

            weighted_graph = predict_weights(g, np.abs(emb_to_use))
            part = CLO.leiden_max_modularity_part_weighted(weighted_graph, trials=100)

            #calculcate the score
            s = PPS.calc_esim(labels, part.membership, normalize=True)
            print(mixing_rate,s)
            score.append(s)

        f.write(f'{mixing_rate} {sum(score) / len(score)}\n')




    f.close()






