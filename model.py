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
        max_len: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.lstm = nn.LSTM(input_size=embeddings.shape[1], hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional,
                               dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(hidden_size * (2 if bidirectional else 1) * num_layers, hidden_size)
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
        #outputs = self.drop(encodings))
        outputs = torch.cat([states[:, 0, :], states[:, -1, :]], dim=1)
        outputs = self.relu(self.linear1(outputs))
        outputs = self.drop(outputs)
        outputs = self.linear2(outputs)
        return outputs


