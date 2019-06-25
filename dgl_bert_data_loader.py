import dgl
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
import numpy as np


class SentPairClsDataLoader(DataLoader):
    def __init__(self,
                 input_ids,
                 segment_ids,
                 input_mask,
                 labels,
                 batch_size,
                 sampler):
        attention_mask = (1. - input_mask) * -1e5

        dataset = TensorDataset(input_ids, segment_ids, attention_mask, labels)
        data_sampler = sampler(dataset)

        super(SentPairClsDataLoader, self).__init__(dataset=dataset,
                                                    batch_size=batch_size,
                                                    sampler=data_sampler,
                                                    collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        graphs = []
        input_ids_list = []
        segment_ids_list = []
        label_list = []
        for input_ids, segment_ids, attention_mask, labels in batch:
            g = dgl.DGLGraph()
            g.add_nodes(input_ids.size(0))
            g.ndata['attention_mask'] = attention_mask

            for i in range(input_ids.size(0)):
                g.add_edges(i, range(input_ids.size(0)))
            graphs.append(g)
            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            label_list.append(labels)

        batch_graph = dgl.batch(graphs)
        return batch_graph, torch.stack(input_ids_list, 0), \
               torch.stack(segment_ids_list, 0), torch.stack(label_list, 0)


if __name__ == "__main__":
    train_input_ids = np.ones([50, 32])
    train_segment_ids = np.ones([50, 32])
    train_input_mask = np.ones([50, 32])
    train_labels = np.ones([50])

    d_loader = SentPairClsDataLoader(train_input_ids,
                                     train_segment_ids,
                                     train_input_mask,
                                     train_labels,
                                     batch_size=5,
                                     sampler=RandomSampler)

    for g, input_ids, segment_ids, labels in d_loader:
        # print(g)
        # print(input_ids.size())
        # print(segment_ids.size())
        # print(labels.size())
        print(g.ndata['attention_mask'].size())
        break

