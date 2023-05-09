import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
from torch_sparse import coalesce
from torch_geometric.utils import to_undirected
from ogb.linkproppred import PygLinkPropPredDataset

from utils import extract_enclosing_subgraphs


def load_data(dataset="ogbl-collab", use_valedges_as_input=True, directed=False):
    dataset = PygLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split()  # get the split edge index
    data = dataset[0]  # train data

    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        if not directed:
            val_edge_index = to_undirected(val_edge_index)
        data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)  # To ensuer the correctness of coalesce
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)  # TODO We will not use edge weights
    return dataset, split_edge, data


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    """
    :param split: the dataset of split target
    :param split_edge: the ogb object, which stores the split edge index
    :param edge_index: the total edge_index of dataset
    :param num_nodes:
    :param percent: the subsample percent
    :return:
    """
    pos_edge = split_edge[split]['edge'].t()  # only the train/valid/test edge. Note that this edge is not doubled?.

    # We don't use the return of edge weight (It is None)
    # Add self loops, the edge_index will be flipped and concatenate to the original edge_index
    new_edge_index, _ = add_self_loops(edge_index)  # To ensure the negative sample will not sample the self loop edge?
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge.size(
            1))  # NOTE: for undirected graph, here the edge_index should be doubled TODO Whether the neg is doubled or not?

    # subsample for pos_edge
    np.random.seed(123)
    num_pos = pos_edge.size(1)  # get the number of pos edges
    perm = np.random.permutation(num_pos)  # get the list of random indexes
    perm = perm[:int(percent / 100 * num_pos)]  # The range of percent is 1-100
    pos_edge = pos_edge[:, perm]

    # subsample for neg_edge
    np.random.seed(123)
    num_neg = neg_edge.size(1)
    perm = np.random.permutation(num_neg)
    perm = perm[:int(percent / 100 * num_neg)]
    neg_edge = neg_edge[:, perm]  # The number is equal to positive samples
    return pos_edge, neg_edge


from torch_geometric.data import InMemoryDataset  # feed to cpu
import scipy.sparse as ssp


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', directed=False):
        """
        :param root: Where the dataset is saved
        :param data: torch_geometric.data.Data
        :param split_edge: Dict ogb data object
        :param num_hops:
        :param percent:
        :param split: train/valid/test
        :param use_coalesce:
        :param node_label:
        :param directed:
        """
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property  # the attributes to be activated
    def processed_file_names(self):
        # Initiation will check whether the file of root\\name is exists
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += ".pt"
        return [name]

    def process(self):  # the initiate of Dataset will automatically call this function
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)
        if 'edge_weight' in self.data:  # reshape edge_weight for transforming to sprase matrix
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)  # The sparse matrix, input value, index_0, index_1, shape
        )

        pos_list = extract_enclosing_subgraphs(pos_edge, A, self.data.x, 1, self.num_hops, directed=self.directed)
        neg_list = extract_enclosing_subgraphs(neg_edge, A, self.data.x, 0, self.num_hops, directed=self.directed)

        temp = self.collate(
            pos_list + neg_list)  # Just save the python list of `torch_geometric.data.Data` as torch_geometric.data.InMemoryDataset object
        torch.save(self.collate(pos_list + neg_list),
                   self.processed_paths[0])  # processed_path is a property of Dataset
        del pos_list, neg_list


def train():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.edge_index, data.batch, data.x)
        loss = criterion(out.view(-1), data.y.to(torch.float))  # Original y is integer, but the out are continuous.
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def test():
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader):
        data = data.to(device)
        out = model(data.z, data.edge_index, data.batch, data.x)
        y_pred.append(out.view(-1).cpu())  # OGB requires the evaluator happens in cpu ?
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true == 1]  # Get the prediction for true positive
    neg_val_pred = val_pred[val_true == 0]  # Get the prediction for true negative

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        out = model(data.z, data.edge_index, data.batch, data.x)
        y_pred.append(out.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]

    results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    return results


if __name__ == "__main__":
    dataset, split_edge, data = load_data()
    path = dataset.root + '_seal'  # The dataset root should be dataset\\ogbl_collab
    train_dataset = SEALDataset(path,
                                data,
                                split_edge,
                                num_hops=1,
                                percent=5,
                                split='train',
                                use_coalesce=True,
                                node_label='drnl',
                                directed=False)
    val_dataset = SEALDataset(path,
                              data,
                              split_edge,
                              num_hops=1,
                              percent=5,
                              split='valid',
                              use_coalesce=True,
                              node_label='drnl',
                              directed=False)
    test_dataset = SEALDataset(path,
                               data,
                               split_edge,
                               num_hops=1,
                               percent=5,
                               split='test',
                               use_coalesce=True,
                               node_label='drnl',
                               directed=False)

    from torch_geometric.data import DataLoader

    batch_size = 32
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers)

    from torch.nn import BCEWithLogitsLoss
    from models import DGCNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(hidden_channels=32, num_layers=3, max_z=1000, k=25, train_dataset=train_dataset).to(
        device)  # To put parameters on same device
    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=list(model.parameters()), lr=0.1)

    from ogb.linkproppred import Evaluator

    evaluator = Evaluator(name="ogbl-collab")

    for epoch in range(1, 100 + 1):
        loss = train()
        results = test()
        for key, result in results.items():
            valid_res, test_res = result
            to_print = (f'Epoch: {epoch:02d}, ' +
                        f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                        f'Test: {100 * test_res:.2f}%')
            print(key)
            print(to_print)
