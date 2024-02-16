import graph_tool.all as gt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
#import sparse
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




#
# Evaluation
#
def calc_esim(y, ypred, normalize=False):
    """
    Element centric similarity.

    y: numpy array of the true group memberships
    ypred: numpy array of the detected group memberships
    normalize: normalize = False gives the element-centric similarity. With normalize=True, the similarity is shifted to be zero for random partition on average and rescaled such that the maximum value is 1.

    """
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
    S = np.sum(np.multiply(Q, (nAB ** 2))) / N

    if normalize == False:
        return S

    # Calc the expected element-centric similarity for random partitions
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    Srand = np.sum(np.multiply(Q, (nAB_rand ** 2))) / N
    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected