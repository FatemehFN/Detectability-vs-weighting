import networkx as nx
import igraph as ig
import leidenalg
import numpy as np
from heapq import heappush


def leiden_partitions(G, reweight=False, n_of_iterations=0):

    U = nx.Graph()
    U.add_nodes_from(sorted(G.nodes(data=True)))
    U.add_edges_from(G.edges(data=True))

    # print(U.nodes(data=True))

    mapping = {}  # from original node names to igraph node names
    rev_mapping = {}  # from igraph node names to original node names
    i = 0
    for node in U.nodes():
        mapping.update({node: i})
        rev_mapping.update({i: node})
        i += 1

    g = ig.Graph(directed=False)
    g.add_vertices(len(U.nodes()))
    for edge in U.edges(data=True):
        g.add_edge(mapping[edge[0]], mapping[edge[1]])

    if reweight == True:
        iter = 1
        while iter <= n_of_iterations:
            # print('here')
            for edge in g.es:
                source_node_index = edge.source
                target_node_index = edge.target
                neighbors_s = set(g.neighbors(source_node_index))
                neighbors_t = set(g.neighbors(target_node_index))

                # print('\n source and target node info')
                # print(source_node_index)
                # print(neighbors_s)

                # print(target_node_index)
                # print(neighbors_t)

                intersection = neighbors_s.intersection(neighbors_t)
                union = neighbors_s.union(neighbors_t)

                intersection_l = len(intersection)
                union_l = len(union)
                if iter == 1:
                    jaccard_index = (intersection_l + 1) / union_l

                else:
                    weight_intersection = 0
                    weight_union = 0
                    for item in intersection:
                        edge_id_s = g.get_eid(source_node_index, item)
                        edge_id_t = g.get_eid(target_node_index, item)

                        # Use the edge_id to access the edge
                        edge_s = g.es[edge_id_s]
                        edge_t = g.es[edge_id_t]
                        weight_intersection += edge_s["weight"]
                        weight_intersection += edge_t["weight"]

                    for item_1 in neighbors_s:
                        edge_id_s_u = g.get_eid(source_node_index, item_1)
                        edge_s_u = g.es[edge_id_s_u]
                        weight_union += edge_s_u["weight"]

                    for item_2 in neighbors_t:
                        edge_id_t_u = g.get_eid(target_node_index, item_2)
                        edge_t_u = g.es[edge_id_t_u]
                        weight_union += edge_t_u["weight"]

                    if weight_union == 0:
                        jaccard_index = 0

                    else:
                        jaccard_index = (weight_intersection + 1) / weight_union

                    print("iter and weight intersection and weight union")
                    print(iter)
                    print(weight_intersection)
                    print(weight_union)
                edge["weight"] = jaccard_index

                # edge['weight'] = 1

                print("edge")
                print(rev_mapping[edge.source])
                print(rev_mapping[edge.target])
                print(edge.attributes())
            iter += 1

        part = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights="weight"
        )
        # part = leidenalg.RBConfigurationVertexPartition(g, weights='weight', resolution_parameter=0.1)
        # part = leidenalg.RBConfigurationVertexPartition(g)
        modularity_value = part.quality()
        print("modularity", modularity_value)
        print(part)
    else:

        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        # part=[[0,1,2,3,4],[5,6,7,8,9]]
        modularity_value = part.quality()
        print("modularity", modularity_value)
        print(part)

    # print(len(part))
    part_in_original_node_names = []
    for node_1 in U.nodes(data=True):
        for p in range(len(part)):
            if mapping[node_1[0]] in part[p]:
                part_in_original_node_names.append(p)

    print("leiden alg done with %s clusters" % (len(part)))

    # print('membership array using leiden')
    # print(len(part_in_original_node_names))
    # print(max(part_in_original_node_names))

    array_part = np.array(part_in_original_node_names)
    print("array part")
    print(array_part)

    return array_part, len(part)

    # return part


def leiden_partitions_fixed_ratio(
    G, ratio, memberships, reweight=False, n_of_iterations=0
):

    U = nx.Graph()
    U.add_nodes_from(sorted(G.nodes(data=True)))
    U.add_edges_from(G.edges(data=True))

    # print(U.nodes(data=True))

    mapping = {}  # from original node names to igraph node names
    rev_mapping = {}  # from igraph node names to original node names
    i = 0
    for node in U.nodes():
        mapping.update({node: i})
        rev_mapping.update({i: node})
        i += 1

    g = ig.Graph(directed=False)
    g.add_vertices(len(U.nodes()))
    for edge in U.edges(data=True):
        g.add_edge(mapping[edge[0]], mapping[edge[1]])

    if reweight == True:
        iter = 1
        while iter <= n_of_iterations:
            # print('here')
            for edge in g.es:
                source_node_index = edge.source
                target_node_index = edge.target
                if memberships[source_node_index] == memberships[target_node_index]:
                    edge["weight"] = ratio
                else:
                    edge["weight"] = 1

                # edge['weight'] = 1

                print("edge")
                print(edge.source)
                print(edge.target)
                print(edge.attributes())
            iter += 1

        part = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights="weight"
        )
        # part = leidenalg.RBConfigurationVertexPartition(g, weights='weight', resolution_parameter=0.1)
        # part = leidenalg.RBConfigurationVertexPartition(g)
        # modularity_value = part.quality()
        # print('modularity',modularity_value)
        # print(part)
    else:

        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        # part=[[0,1,2,3,4],[5,6,7,8,9]]
        # modularity_value = part.quality()
        # print('modularity', modularity_value)
        # print(part)

    # print(len(part))
    part_in_original_node_names = []
    for node_1 in U.nodes(data=True):
        for p in range(len(part)):
            if mapping[node_1[0]] in part[p]:
                part_in_original_node_names.append(p)

    print("leiden alg done with %s clusters" % (len(part)))

    # print('membership array using leiden')
    # print(len(part_in_original_node_names))
    # print(max(part_in_original_node_names))

    array_part = np.array(part_in_original_node_names)
    # print('array part')
    # print(array_part)

    # calculate mu for all nodes
    for node in g.vs:
        node["mu"] = mixing_parameter(node, g, memberships)
        print(node["mu"])

    sum_mu = 0
    for n in range(500):
        sum_mu += g.vs[n]["mu"]

    print(sum_mu / 500)

    return array_part, len(part)

    # return part


def mixing_parameter(node, g, memberships):

    k_in = 0
    k = 0

    for item in g.vs:
        if g.are_connected(node, item):
            edge_index = g.get_eid(node, item)
            if memberships[item.index] == memberships[node.index]:

                k_in += g.es[edge_index]["weight"]

            k += g.es[edge_index]["weight"]

    return 1 - (k_in / k)


def leiden_max_modularity_part_weighted(g, trials=5):
    maxModularity = -1
    part_to_return = None
    weights = [edge["weight"] for edge in g.es]
    for _ in range(trials):
        part = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights=weights
        )
        modularity = part.quality()

        if modularity > maxModularity:
            maxModularity = modularity
            part_to_return = part

    return part_to_return


def leiden_max_modularity_part(g, trials=5):
    maxModularity = -1
    part_to_return = None
    for _ in range(trials):
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        modularity = part.quality()

        if modularity > maxModularity:
            maxModularity = modularity
            part_to_return = part

    return part_to_return


def leiden_partitions_bc(G, reweight=False, n_of_iterations=0):

    U = nx.Graph()
    U.add_nodes_from(sorted(G.nodes(data=True)))
    U.add_edges_from(G.edges(data=True))

    # print(U.nodes(data=True))

    mapping = {}  # from original node names to igraph node names
    rev_mapping = {}  # from igraph node names to original node names
    i = 0
    for node in U.nodes():
        mapping.update({node: i})
        rev_mapping.update({i: node})
        i += 1

    # print('mapping and rev mapping')
    # print(mapping)
    # print(rev_mapping)

    g = ig.Graph(directed=False)
    g.add_vertices(len(U.nodes()))
    for edge in U.edges(data=True):
        g.add_edge(mapping[edge[0]], mapping[edge[1]])

    edge_betweenness = g.edge_betweenness(directed=False)
    # print(edge_betweenness)

    if reweight == True:
        iter = 1
        while iter <= n_of_iterations:
            # print('here')
            for edge in g.es:
                edge_index = g.get_eid(edge.source, edge.target, directed=False)
                edge["weight"] = 1 / (edge_betweenness[edge_index]) ** 2
                # edge['weight'] = 1 / (edge_betweenness[edge_index])

                # print('edge')
                # print(edge.source)
                # print(edge.target)
                # print(edge.attributes())
            iter += 1

        part = leiden_max_modularity_part_weighted(g, trials=100)
        # part = leidenalg.RBConfigurationVertexPartition(g, weights='weight', resolution_parameter=0.1)
        # part = leidenalg.RBConfigurationVertexPartition(g)

    else:

        part = leiden_max_modularity_part(g, trials=100)
        # part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        # part=[[0,1,2,3,4],[5,6,7,8,9]]

    # print(len(part))
    part_in_original_node_names = []
    for node_1 in U.nodes(data=True):
        for p in range(len(part)):
            if mapping[node_1[0]] in part[p]:
                part_in_original_node_names.append(p)

    print("leiden alg done with %s clusters" % (len(part)))

    # print('membership array using leiden')
    # print(len(part_in_original_node_names))
    # print(max(part_in_original_node_names))

    array_part = np.array(part_in_original_node_names)
    # print('array part')
    # print(array_part)

    return array_part, len(part)

    # return part


def leiden_partitions_PPR(G, reweight=False, n_of_iterations=0):

    U = nx.Graph()
    U.add_nodes_from(sorted(G.nodes(data=True)))
    U.add_edges_from(G.edges(data=True))

    # print(U.nodes(data=True))

    mapping = {}  # from original node names to igraph node names
    rev_mapping = {}  # from igraph node names to original node names
    i = 0
    for node in U.nodes():
        mapping.update({node: i})
        rev_mapping.update({i: node})
        i += 1

    # print('mapping and rev mapping')
    # print(mapping)
    # print(rev_mapping)

    g = ig.Graph(directed=False)
    g.add_vertices(len(U.nodes()))
    for edge in U.edges(data=True):
        g.add_edge(mapping[edge[0]], mapping[edge[1]])

    if reweight == True:
        iter = 1
        while iter <= n_of_iterations:
            # print('here')
            for edge in g.es:

                edge["weight"] = personalized_pagerank_similarity(
                    g, edge.source, edge.target
                )

                # print('edge')
                # print(edge.source)
                # print(edge.target)
                # print(edge.attributes())
            iter += 1

        part = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights="weight"
        )
        # part = leidenalg.RBConfigurationVertexPartition(g, weights='weight', resolution_parameter=0.1)
        # part = leidenalg.RBConfigurationVertexPartition(g)

    else:

        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        # part=[[0,1,2,3,4],[5,6,7,8,9]]

    # print(len(part))
    part_in_original_node_names = []
    for node_1 in U.nodes(data=True):
        for p in range(len(part)):
            if mapping[node_1[0]] in part[p]:
                part_in_original_node_names.append(p)

    print("leiden alg done with %s clusters" % (len(part)))

    # print('membership array using leiden')
    # print(len(part_in_original_node_names))
    # print(max(part_in_original_node_names))

    array_part = np.array(part_in_original_node_names)
    # print('array part')
    # print(array_part)

    return array_part, len(part)

    # return part


def reweight_adj_matrix(adj_matrix, G, n_of_iterations=1):

    U = nx.Graph()
    U.add_nodes_from(sorted(G.nodes(data=True)))
    U.add_edges_from(G.edges(data=True))

    # print(U.nodes(data=True))

    mapping = {}  # from original node names to igraph node names
    rev_mapping = {}  # from igraph node names to original node names
    i = 0
    for node in U.nodes():
        mapping.update({node: i})
        rev_mapping.update({i: node})
        i += 1

    g = ig.Graph(directed=False)
    g.add_vertices(len(U.nodes()))
    for edge in U.edges(data=True):
        g.add_edge(mapping[edge[0]], mapping[edge[1]])

    iter = 1

    while iter <= n_of_iterations:
        # print('here')
        for edge in g.es:
            source_node_index = edge.source
            target_node_index = edge.target
            neighbors_s = set(g.neighbors(source_node_index))
            neighbors_t = set(g.neighbors(target_node_index))

            # print('\n source and target node info')
            # print(source_node_index)
            # print(neighbors_s)

            # print(target_node_index)
            # print(neighbors_t)

            intersection = neighbors_s.intersection(neighbors_t)
            union = neighbors_s.union(neighbors_t)

            intersection_l = len(intersection)
            union_l = len(union)
            if iter == 1:
                jaccard_index = (intersection_l + 1) / union_l

            else:
                weight_intersection = 0
                weight_union = 0
                for item in intersection:
                    edge_id_s = g.get_eid(source_node_index, item)
                    edge_id_t = g.get_eid(target_node_index, item)

                    # Use the edge_id to access the edge
                    edge_s = g.es[edge_id_s]
                    edge_t = g.es[edge_id_t]
                    weight_intersection += edge_s["weight"]
                    weight_intersection += edge_t["weight"]

                for item_1 in neighbors_s:
                    edge_id_s_u = g.get_eid(source_node_index, item_1)
                    edge_s_u = g.es[edge_id_s_u]
                    weight_union += edge_s_u["weight"]

                for item_2 in neighbors_t:
                    edge_id_t_u = g.get_eid(target_node_index, item_2)
                    edge_t_u = g.es[edge_id_t_u]
                    weight_union += edge_t_u["weight"]

                if weight_union == 0:
                    jaccard_index = 0

                else:
                    jaccard_index = (weight_intersection + 1) / weight_union

                # print('iter and weight intersection and weight union')
                # print(iter)
                # print(weight_intersection)
                # print(weight_union)
            edge["weight"] = jaccard_index
            adj_matrix[source_node_index, target_node_index] = jaccard_index
            adj_matrix[target_node_index, source_node_index] = jaccard_index

            # edge['weight'] = 1

            # print(edge.attributes())
        iter += 1

    # print(adj_matrix)
    return adj_matrix


def personalized_pagerank_similarity(
    graph, node1, node2, damping=0.85, max_iter=100, tol=1e-6
):
    """
    Calculate the Personalized PageRank similarity between two nodes in a graph.

    Parameters:
    - graph: igraph Graph
    - node1, node2: Nodes for which similarity is calculated
    - damping: Damping factor (typically between 0.8 and 0.9)
    - max_iter: Maximum number of iterations for power iteration method
    - tol: Tolerance to declare convergence

    Returns:
    - Similarity score between node1 and node2
    """

    # Create a personalized teleport probability vector
    teleport = [0] * graph.vcount()
    teleport[node1] = 1  # Teleport to node1

    # Power iteration method to calculate PageRank
    pr1 = graph.personalized_pagerank(weights=None, reset=teleport)

    teleport = [0] * graph.vcount()
    teleport[node2] = 1  # Teleport to node2

    pr2 = graph.personalized_pagerank(weights=None, reset=teleport)

    # Calculate cosine similarity between the two PageRank vectors
    similarity = sum(x * y for x, y in zip(pr1, pr2)) / (
        sum(x**2 for x in pr1) ** 0.5 * sum(x**2 for x in pr2) ** 0.5
    )

    return similarity


def leiden_partitions_walktrap(G, t=3):

    U = nx.Graph()
    U.add_nodes_from(sorted(G.nodes(data=True)))
    U.add_edges_from(G.edges(data=True))

    mapping = {}  # from original node names to igraph node names
    rev_mapping = {}  # from igraph node names to original node names
    i = 0
    for node in U.nodes():
        mapping.update({node: i})
        rev_mapping.update({i: node})
        i += 1

    g = ig.Graph(directed=False)
    g.add_vertices(len(U.nodes()))
    for edge in U.edges(data=True):
        g.add_edge(mapping[edge[0]], mapping[edge[1]])

    weighted_g = walktrap_edge_weights(g, t=t)

    # for edge in weighted_g.es:
    # print(f"Edge {edge.source} - {edge.target}: Weight = {edge['weight']}")

    part = leiden_max_modularity_part_weighted(weighted_g, trials=100)

    part_in_original_node_names = []
    for node_1 in U.nodes(data=True):
        for p in range(len(part)):
            if mapping[node_1[0]] in part[p]:
                part_in_original_node_names.append(p)

    print("leiden alg done with %s clusters" % (len(part)))

    # print('membership array using leiden')
    # print(len(part_in_original_node_names))
    # print(max(part_in_original_node_names))

    array_part = np.array(part_in_original_node_names)
    # print('array part')
    # print(array_part)

    return array_part, len(part)


def walktrap_edge_weights(graph, t):
    def walktrap_similarity(C1, C2, Dx, P_t, N):
        delta_sigma_C1C2 = (0.5 / N) * np.sum(
            np.square(np.matmul(Dx, P_t[C1]) - np.matmul(Dx, P_t[C2]))
        )
        return delta_sigma_C1C2

    G = graph.copy()

    N = G.vcount()
    A = np.array(G.get_adjacency().data)
    Dx = np.zeros((N, N))
    P = np.zeros((N, N))

    for i, A_row in enumerate(A):
        d_i = np.sum(A_row)
        P[i] = A_row / d_i
        Dx[i, i] = d_i ** (-0.5)

    P_t = np.linalg.matrix_power(P, t)

    min_sigma_heap = []
    for edge in G.es:
        C1_id = edge.source
        C2_id = edge.target

        ds = walktrap_similarity(C1_id, C2_id, Dx, P_t, N)
        heappush(min_sigma_heap, (ds, C1_id, C2_id))

    # Assign edge weights to the graph
    for edge in G.es:
        weight = min_sigma_heap.pop(0)[0]
        edge["weight"] = weight

    return G
