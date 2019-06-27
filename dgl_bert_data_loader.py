import dgl
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class ListDataset(Dataset):
    def __init__(self, *lists_of_data):
        assert all(len(lists_of_data[0]) == len(d) for d in lists_of_data)
        self.lists_of_data = lists_of_data

    def __getitem__(self, index):
        return tuple(d[index] for d in self.lists_of_data)

    def __len__(self):
        return len(self.lists_of_data[0])


class SentPairClsDataLoader(DataLoader):
    def __init__(self,
                 input_ids,
                 segment_ids,
                 labels,
                 batch_size,
                 sampler,
                 device=None,
                 label_dtype=None):
        dataset = ListDataset(input_ids, segment_ids, labels)
        data_sampler = sampler(dataset)

        self.device = device
        self.label_dtype = label_dtype

        super(SentPairClsDataLoader, self).__init__(dataset=dataset,
                                                    batch_size=batch_size,
                                                    sampler=data_sampler,
                                                    collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        graphs = []
        label_list = []
        for input_ids, segment_ids, labels in batch:
            seq_length = len(input_ids)
            g = dgl.DGLGraph()
            g.add_nodes(seq_length)
            g.ndata['input_ids'] = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            g.ndata['segment_ids'] = torch.tensor(segment_ids, dtype=torch.long, device=self.device)
            g.ndata['position_ids'] = torch.arange(len(input_ids), dtype=torch.long, device=self.device)

            for i in range(seq_length):
                g.add_edges(i, range(seq_length))
            graphs.append(g)
            label_list.append(labels)

        batch_graph = dgl.batch(graphs)
        return batch_graph, torch.tensor(label_list, dtype=self.label_dtype, device=self.device)


if __name__ == "__main__":
    train_input_ids = []
    train_input_ids.append(np.ones(5))
    train_input_ids.append(np.ones(15))
    train_input_ids.append(np.ones(10))

    train_segment_ids = []
    train_segment_ids.append(np.ones(5))
    train_segment_ids.append(np.ones(15))
    train_segment_ids.append(np.ones(10))

    train_labels = np.ones(3)

    d_loader = SentPairClsDataLoader(train_input_ids,
                                     train_segment_ids,
                                     train_labels,
                                     batch_size=3,
                                     sampler=RandomSampler)

    for g, labels in d_loader:
        print(g)
        print(g.ndata)
        print(g.ndata['input_ids'].size())
        print(labels.size())

