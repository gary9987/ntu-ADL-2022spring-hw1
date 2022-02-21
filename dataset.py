from typing import List, Dict

from torch.utils.data import Dataset
import torch
from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    '''
    def __getitem__(self, index) -> Dict:
    instance = self.data[index]
    return instance
    '''
    def __getitem__(self, index):
        instance = self.data[index]
        str = instance['text']
        try:
            label = instance['intent']
            label = self.label2idx(label)
        except:
            label = ""

        encoding = self.vocab.encode_batch([str], to_len=self.max_len)
        encoding = torch.FloatTensor(encoding).view(-1)
        return torch.FloatTensor(encoding), label

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    '''
    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        raise NotImplementedError
    '''
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
