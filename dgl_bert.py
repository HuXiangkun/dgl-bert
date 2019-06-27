from torch import nn
import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertLayerNorm, \
    BertIntermediate, BertOutput, BertSelfOutput, BertConfig
import copy
import math
import numpy as np
from dgl_bert_data_loader import SentPairClsDataLoader
from torch.utils.data import RandomSampler
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, graph):
        node_num = graph.ndata['h'].size(0)

        Q = self.query(graph.ndata['h'])
        K = self.key(graph.ndata['h'])
        V = self.value(graph.ndata['h'])

        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        graph.ndata['Q'] = Q
        graph.ndata['K'] = K
        graph.ndata['V'] = V

        graph.apply_edges(fn.u_mul_v('K', 'Q', 'attn_probs'))
        graph.edata['attn_probs'] = graph.edata['attn_probs'].sum(-1, keepdim=True)
        graph.edata['attn_probs'] = edge_softmax(graph, graph.edata['attn_probs'])
        graph.edata['attn_probs'] = self.dropout(graph.edata['attn_probs'])
        graph.apply_edges(fn.u_mul_e('V', 'attn_probs', 'attn_values'))

        graph.register_message_func(fn.copy_e('attn_values', 'm'))
        graph.register_reduce_func(fn.sum('m', 'h'))
        graph.update_all()
        graph.ndata['h'] = graph.ndata['h'].view([node_num, -1])

        return graph


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, graph):
        input_tensor = graph.ndata['h']
        self_output_graph = self.self(graph)
        attention_output = self.output(self_output_graph.ndata['h'], input_tensor)
        graph.ndata['h'] = attention_output
        return graph


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, graph):
        graph = self.attention(graph)
        intermediate_output = self.intermediate(graph.ndata['h'])
        layer_output = self.output(intermediate_output, graph.ndata['h'])
        graph.ndata['h'] = layer_output
        return graph


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, graph):
        for layer_module in self.layer:
            graph = layer_module(graph)
        return graph


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, graph):
        embedding_output = self.embeddings(graph.ndata['input_ids'],
                                           graph.ndata['position_ids'],
                                           graph.ndata['segment_ids'])

        graph.ndata.pop('input_ids')
        graph.ndata.pop('position_ids')
        graph.ndata.pop('segment_ids')

        hidden_size = embedding_output.size(-1)
        embedding_output = embedding_output.view(-1, hidden_size)

        graph.ndata['h'] = embedding_output

        graph = self.encoder(graph)

        g_list = dgl.unbatch(graph)

        pooled_output = []
        for g in g_list:
            pooled_output.append(g.ndata['h'][0])
        pooled_output = torch.stack(pooled_output, 0)

        pooled_output = self.pooler(pooled_output)
        return graph, pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, graph, labels=None):
        _, pooled_output = self.bert(graph)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


if __name__ == '__main__':
    config = BertConfig(vocab_size_or_config_json_file=30522)
    bert = BertModel.from_pretrained('bert-base-uncased')
    # print(bert)

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
        bert(g)
