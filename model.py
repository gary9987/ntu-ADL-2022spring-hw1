from typing import Dict

import torch
from torch.nn import Embedding
from torch import nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.lstm = nn.LSTM(input_size=embeddings.shape[1], hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional,
                               dropout=dropout, batch_first=True)

        self.attention = Attention(hidden_size * (2 if bidirectional else 1) * max_len, max_len)

        #self.linear = nn.Linear(hidden_size*(2 if bidirectional else 1)*max_len, num_class)
        self.linear1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError
    '''
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
    '''
    def forward(self, x):
        x = self.embed(x)
        states, (h, c) = self.lstm(x)
        #encodings = torch.cat([states[:, 0, :], states[:, -1, :]], dim=1)
        #encodings = torch.cat([states[:, i, :] for i in range(states.shape[1])], dim=1)
        #outputs = self.drop(encodings)
        outputs = self.attention(states)
        outputs = self.relu(self.linear1(outputs))
        outputs = self.drop(outputs)
        outputs = self.linear2(outputs)
        return outputs


