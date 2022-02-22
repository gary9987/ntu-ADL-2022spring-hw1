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

        self.drop = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size * 4, num_class)
        self.relu = nn.ReLU()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.embed(x)
        states, _ = self.lstm(x)
        outputs = torch.cat([states[:, 0, :], torch.flip(states[:, -1, :], [1])], dim=1)
        outputs = self.drop(outputs)
        outputs = self.relu(self.linear1(outputs))
        return {'outputs': outputs}
