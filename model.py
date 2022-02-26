from typing import Dict

import torch
from torch.nn import Embedding
from torch import nn


class SeqClassifier(torch.nn.Module):
    def __init__(
            self,
            embeddings: torch.tensor,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
            num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = nn.GRU(input_size=embeddings.shape[1], hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional,
                          dropout=dropout, batch_first=True)

        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size * (2 if bidirectional else 1) * num_layers)
        self.linear1 = nn.Linear(hidden_size * (2 if bidirectional else 1) * num_layers, 512)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, num_class)


    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.embed(x)
        _, h = self.rnn(x)
        h = h.permute(1, 0, 2)
        h = h.reshape(h.shape[0], -1)
        outputs = self.drop(h)
        outputs = self.bn1(outputs)
        outputs = self.relu(self.linear1(outputs))
        outputs = self.drop(self.bn2(outputs))
        outputs = self.relu(self.linear2(outputs))
        return {'outputs': outputs}
