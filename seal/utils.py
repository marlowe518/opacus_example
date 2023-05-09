import torch
import numpy as np

from scipy.sparse.csgraph import shortest_path
import numpy as np


def drnl_node_labeling(adj, src, dst):
    """
    :param adj: The adjacency matrix of subgraph
    :param src: The node index of src of target link
    :param dst: The node index of dst of target link
    :return:
    """
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)  # Ensure the index of src is smaller than dst

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]  # get the adjacency matrix without src

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:,
                 idx]  # get the adjacency matrix without dst. Note this will result some node become unreachable, and lead to infinite distance

    # Get the distance from each node to src
    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)  # src to src is 0
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                             indices=dst - 1)  # The index of dst should -1 since it is located after src (the deleted row/col)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    # Compute according to the Eq.(10) in the original paper.
    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)

from torch_geometric.data import Data
import scipy.sparse as ssp


def construct_pyg_graph(node_ids, adj, node_features, y):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, _ = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)  # TODO
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    # r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    # edge_weight = r.to(torch.float) TODO we do not need edge weight
    y = torch.tensor([y])
    z = drnl_node_labeling(adj, 0, 1)
    # data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
    #             node_id=node_ids, num_nodes=num_nodes)
    data = Data(node_features, edge_index, y=y, z=z,
                node_id=node_ids, num_nodes=num_nodes)
    return data

def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A,
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)  # A[i,j] will only return non-zero elements.
    else:
        res = set(A[:, list(fringe)].indices)

    return res

def k_hop_subgraph(src, dst, num_hops, A, node_features=None,
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops + 1):
        if not directed:
            fringe = neighbors(fringe, A)  # Doubled A already includes out and in neighbors
        else:
            pass
            # out_neighbors = neighbors(fringe, A)
            # in_neighbors = neighbors(fringe, A_csc, False)
            # fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited  # a-b, set a will delete the elements in b; Delete the elements are already collected neighbors
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)  # record the distance of each node
    subgraph = A[nodes, :][:, nodes]  # extract the node related edges

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0  # the new matrix is permuted by subscription.
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y

from tqdm import tqdm


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        nodes, subgraph, _, node_features, y = k_hop_subgraph(src, dst, num_hops, A, node_features=x, y=y,
                                                              directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(nodes, subgraph, node_features, y)
        data_list.append(data)

    return data_list
