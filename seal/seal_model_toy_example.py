import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


def load_data():
    torch.manual_seed(0)

    # create edges between nodes (we have 15 nodes in this graph)
    edge_index = torch.tensor([[0, 2, 3, 5, 5, 7, 8, 10, 11, 12, 13, 14, 14, 14],
                               [1, 1, 2, 4, 6, 8, 9, 9, 10, 10, 12, 11, 9, 8]])
    edge_index = torch.cat([edge_index, edge_index.flip(0)],
                           1)  # change to undirected graph: flip alone the axis 0, the src and dst will filped, and concate to build another direction edges.
    print(f"edge_index.size:{edge_index.size()}")
    print(f"edge_index:\n{edge_index.T.data}")

    # features (just random here)
    n_feats = 10
    x = torch.randint(low=0, high=4, size=(edge_index.unique().size(0), n_feats))
    """
    .unique(): get the unique elements in the tensor
    size(): # nodes
    the features values is randint in [0, 3]
    """

    # train, test, val masks for each node # sequence of 0, 1, just computing according to the split rate below
    train_mask = torch.tensor([True] * round(edge_index.unique().size(0) * 0.8) +
                              [False] * (edge_index.unique().size(0) - round(edge_index.unique().size(0) * 0.8)))
    test_mask = torch.tensor([False] * round(edge_index.unique().size(0) * 0.8) +
                             [True] * (round(edge_index.unique().size(0) * 0.1)) +
                             [False] * (edge_index.unique().size(0) - round(edge_index.unique().size(0) * 0.8)
                                        - round(edge_index.unique().size(0) * 0.1)))
    val_mask = torch.tensor([False] * round(edge_index.unique().size(0) * 0.8) +
                            [False] * (round(edge_index.unique().size(0) * 0.1)) +
                            [True] * (edge_index.unique().size(0) - round(edge_index.unique().size(0) * 0.8)
                                      - round(edge_index.unique().size(0) * 0.1)))  # TODO why mask nodes?

    new_data = Data(edge_index=edge_index,
                    x=x,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask)
    print_data(new_data)
    # attributes including: 1. x(features), 2. y, 3. edge_index with shape (2, edge nums);
    # The additional keys are in attr of node stores and stores

    # plot the graph
    draw_graph(edge_index)
    return new_data


def print_data(data: torch_geometric.data.Data, tag="DATA LOADED"):
    print("*" * 80 + f"{tag} OVERVIEW" + "*" * 80)
    print(*[f"{key}:{val.size()}" for key, val in data.items()], sep="\n")
    print("*" * 80 + f"{tag}" + "*" * 80)
    print(*[f"{key}\n{val}" for key, val in data.items()], sep="\n")


def draw_graph(edge_index):
    G = nx.Graph()
    G.add_edges_from([(r1, r2) for r1, r2 in zip(edge_index.numpy()[0], edge_index.numpy()[1])])
    nx.draw(G, cmap=plt.get_cmap('jet'), with_labels=True)
    plt.show()


# source : https://github.com/rusty1s/pytorch_geometric/blob/master/examples/seal_link_pred.py
import math
import os.path as osp
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, global_sort_pool
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix


def extract_enclosing_subgraphs(df, link_index, edge_index, num_hops, y):
    """
    Extract enclosing subgraphs for every pair of nodes in the graph and label the nodes
    in each extracted subgraph.

    Args:
    :param link_index : edges in the graph ranxun: TODO link_index should be the undoubellded type
    :param edge_index : edges in the graph
    :param num_hops : number of hops
    :param y : if true link
    """
    data_list = []
    for src, dst in link_index.t().tolist():
        src_origin = src
        dst_origin = dst
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops, edge_index,
            relabel_nodes=True)  # relabel Ture, label index in sub_edge_index will be changed.
        src, dst = mapping.tolist()  # the new ids where src, dst are mapped into?

        # Remove target link from the subgraph.
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)  # 对于每一条边，节点两端的(src,dst)对上了，就为False
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)  # 和(dst,src)对上了，就为False
        sub_edge_index = sub_edge_index[:, mask1 & mask2]  # 两种情况满足一个，就不选中这个边

        # Calculate node labeling.
        z = drnl_node_labeling(sub_edge_index, src, dst,
                               num_nodes=sub_nodes.size(0))

        data = Data(x=df.x[sub_nodes], z=z, src=src_origin, dst=dst_origin,
                    edge_index=sub_edge_index, y=y, sub_nodes=sub_nodes)  # sub nodes are original index.
        data_list.append(data)

    return data_list


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    """
    Label the nodes in each subgraph (z = 1 for source and target nodes)
    assigning a label to each node according to the following formula :
    fl(i) = 1 + min(dx, dy) + (d/2)[(d/2) + (d%2) − 1], (10)
    where dx := d(i, x), dy := d(i, y), d := dx + dy, (d/2) and (d%2) are the integer quotient and
    remainder of d divided by 2, respectively
    Thus the label depends on the distance between the node and the source and target nodes

    Args:
    :param edge_index : edges in the graph
    :param src : source node
    :param src : target node
    :param num_nodes : number of nodes in the subgraph

    """
    global _max_z_
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()  # to sparse adjacency matrix

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True)  # shortest path for between nodes
    dist2src = dist2src[:, src]
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True)
    dist2dst = dist2dst[:, dst - 1]
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    _max_z_ = max(int(z.max()), _max_z_)

    return z.to(torch.long)


def labelling_example(edge_index):
    # Example labelling
    src_orig, dst_orig = 11, 14
    num_hops = 1

    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        [src_orig, dst_orig], num_hops, edge_index, relabel_nodes=True)
    src, dst = mapping.tolist()
    num_nodes = sub_nodes.size(0)

    # Remove target link from the subgraph.
    mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
    mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
    sub_edge_index = sub_edge_index[:, mask1 & mask2]

    z_scores = drnl_node_labeling(sub_edge_index, src, dst, num_nodes)
    print("sub_nodes", sub_nodes)
    print("z_scores", z_scores)

    """ 
    sub_nodes :  tensor([ 7,  8,  9, 10, 11, 12, 13, 14])
    z_scores  :  tensor([1, 4, 5, 4, 7, 1, 0, 7])
    """


from torch_geometric.data import InMemoryDataset
import math
import torch
from torch_geometric.utils import to_undirected, negative_sampling, add_self_loops
import random
import copy


class SEALDataset(InMemoryDataset):
    def __init__(self, df):
        super(SEALDataset, self).__init__()
        self.data, self.slices = df[0], df[1]


def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1, undirected=True):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.
    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)
    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    random.seed(77)
    torch.manual_seed(77)

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]  # get the original edge index (before changing to undirected type)

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))  # return random permutation
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]  # get the row and col edge index
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if undirected:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)  # upper triangular is True, others are False
    neg_adj_mask[row, col] = 0  # edges in whole data is set to 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()  # upper tri elements and not the edges in data
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0  # choose other non-zero elements to zero for neg samples in valid and test
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)  # valid and test does not need to be undirected

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    # draw_graph(data.train_pos_edge_index)

    return data


def process_data(new_data, batch_size):
    random.seed(12345)
    torch.manual_seed(12345)
    num_hops = 1

    # ranxun: split and negative samples, add neg mask/id, pos id to attributes of torch_geometric.Data
    # data_splitted = train_test_split_edges(copy.copy(new_data), val_ratio=0.1, undirected=False)
    data_splitted = train_test_split_edges(copy.copy(new_data), val_ratio=0.1, undirected=True)

    edge_index_loop, _ = add_self_loops(
        data_splitted.train_pos_edge_index)  # change edge index, fill edge attributes when edges are weighted
    data_splitted.train_neg_edge_index = negative_sampling(
        edge_index_loop, num_nodes=data_splitted.num_nodes,
        num_neg_samples=data_splitted.train_pos_edge_index.size(1))  # given value to newly added attributes of Data
    print_data(data_splitted, tag="DATA_SPLITTED")
    # Collect a list of subgraphs for training, validation and test.
    train_pos_list = extract_enclosing_subgraphs(new_data,
                                                 data_splitted.train_pos_edge_index, data_splitted.train_pos_edge_index,
                                                 num_hops, 1)  # TODO why edge_index_loop is not used
    train_neg_list = extract_enclosing_subgraphs(new_data,
                                                 data_splitted.train_neg_edge_index, data_splitted.train_pos_edge_index,
                                                 num_hops, 0)

    val_pos_list = extract_enclosing_subgraphs(new_data,
                                               data_splitted.val_pos_edge_index, data_splitted.train_pos_edge_index,
                                               num_hops, 1)
    val_neg_list = extract_enclosing_subgraphs(new_data,
                                               data_splitted.val_neg_edge_index, data_splitted.train_pos_edge_index,
                                               num_hops, 0)

    test_pos_list = extract_enclosing_subgraphs(new_data,
                                                data_splitted.test_pos_edge_index, data_splitted.train_pos_edge_index,
                                                num_hops, 1)
    test_neg_list = extract_enclosing_subgraphs(new_data,
                                                data_splitted.test_neg_edge_index, data_splitted.train_pos_edge_index,
                                                num_hops, 0)

    for dt in chain(train_pos_list, train_neg_list, val_pos_list,
                    val_neg_list, test_pos_list, test_neg_list):
        print("=" * 80)
        print(f"dt.__class__", dt.__class__.__name__)
        print(f"type(dt)", type(dt))
        print(f"dt:{dt}")
        print(f"dt.src, dt.dst:{dt.src, dt.dst}")
        print(f"dt.z:{dt.z}")
        print(f"dt.x:{dt.x}")
        print(f"dt.edge_index:{dt.edge_index}")
        print(f"dt.subnodes:{dt.sub_nodes}")
        z = F.one_hot(dt.z, _max_z_ + 1).to(torch.float)
        print(f"z:{z}")
        dt.x = torch.cat([z, dt.x], 1)
        print(f"dt.x:{dt.x}")

    # concatenate all the subgraphs
    print("train_pos_list", *train_pos_list, sep='\n')
    train_dataset = InMemoryDataset.collate(train_pos_list + train_neg_list)
    test_dataset = InMemoryDataset.collate(test_pos_list + test_neg_list)
    val_dataset = InMemoryDataset.collate(val_pos_list + val_neg_list)
    print(train_dataset[0], sep="\n")
    print(*train_dataset[1].items(), sep="\n")

    # dataloaders for train, test, val
    train_loader = DataLoader(SEALDataset(train_dataset), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(SEALDataset(val_dataset), batch_size=batch_size)
    test_loader = DataLoader(SEALDataset(test_dataset), batch_size=batch_size)
    return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader


import torch
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d

import math

from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import to_undirected


class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, GNN=torch_geometric.nn.GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_loader])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset[0].num_features, hidden_channels, normalize=False, bias=False))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels, normalize=False, bias=False))
        self.convs.append(GNN(hidden_channels, 1, normalize=False, bias=False))

        conv1d_channels = [4, 8]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0], bias=False)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1, bias=False)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 10)
        self.lin2 = Linear(10, 1)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            conv_out = conv(xs[-1], edge_index)
            activation_out = torch.tanh(conv_out)
            xs += [torch.tanh(conv(xs[-1], edge_index))]

        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(out.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        print(f'loss:{loss.item()}')


if __name__ == "__main__":
    _max_z_ = 0
    batch_size = 1
    new_data = load_data()
    train_dataset, test_dataset, val_dataset, \
        train_loader, test_loader, val_loader = process_data(new_data, batch_size)

    hidden_channel = 5
    num_layers = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(hidden_channels=hidden_channel, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(params=list(model.parameters()), lr=0.01)
    train()