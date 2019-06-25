from torch import nn
import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEmbeddings, BertPooler, \
    BertIntermediate, BertOutput, BertSelfOutput, BertConfig
import copy
import math
import numpy as np
from dgl_bert_data_loader import SentPairClsDataLoader
from torch.utils.data import RandomSampler
import dgl


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
        return x.permute(0, 2, 1, 3)

    def forward(self, graph):
        graph.register_message_func(self.message_func)
        graph.register_reduce_func(self.reduce_func)
        graph.update_all()
        return graph

    def message_func(self, edges):
        return {'h': edges.src['h'],
                'attention_mask': edges.src['attention_mask']}

    def reduce_func(self, nodes):
        hidden_states = nodes.mailbox['h']
        attention_mask = nodes.mailbox['attention_mask']

        mixed_query_layer = self.query(nodes.data['h'])
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer.unsqueeze(1))
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return {'h': context_layer.squeeze(1)}


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


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, graph, input_ids, token_type_ids=None):
        batch_size = input_ids.size(0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        hidden_size = embedding_output.size(-1)
        embedding_output = embedding_output.view(-1, hidden_size)

        graph.ndata['h'] = embedding_output
        graph.ndata['attention_mask'] = graph.ndata['attention_mask'].to(input_ids.device)

        encoded_graph = self.encoder(graph)

        encoded_output = encoded_graph.ndata['h'].view(batch_size, -1, hidden_size)

        pooled_output = self.pooler(encoded_output)
        return encoded_output, pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, graph, input_ids, token_type_ids=None, labels=None):
        _, pooled_output = self.bert(graph, input_ids, token_type_ids)
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
    print(bert)

    # train_input_ids = np.ones([50, 32])
    # train_segment_ids = np.ones([50, 32])
    # train_input_mask = np.ones([50, 32])
    # train_labels = np.ones([50])
    #
    # d_loader = SentPairClsDataLoader(train_input_ids,
    #                                  train_segment_ids,
    #                                  train_input_mask,
    #                                  train_labels,
    #                                  batch_size=5,
    #                                  sampler=RandomSampler)
    #
    # for g, input_ids, segment_ids, labels in d_loader:
    #     bert(g, input_ids, segment_ids)
    #     break
