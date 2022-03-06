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
            task='intent'
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.task = task
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.ln = nn.LayerNorm(embeddings.shape[1])
        self.rnn = nn.GRU(input_size=embeddings.shape[1], hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional,
                          dropout=dropout, batch_first=True)

        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size * (2 if bidirectional else 1))
        self.linear1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 512)
        self.act = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, num_class)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.ln(self.embed(x))
        outputs, h = self.rnn(x)
        if self.task == 'intent':
            outputs = outputs[:, -1, :]
        elif self.task == 'slot':
            outputs = outputs.permute(0, 2, 1)

        outputs = self.bn1(outputs)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.act(self.linear1(outputs))
        outputs = outputs.permute(0, 2, 1)
        outputs = self.bn2(outputs)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.act(self.linear2(outputs))
        return {'outputs': outputs}
