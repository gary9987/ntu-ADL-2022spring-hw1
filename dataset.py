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
            task='intent'
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.task = task

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        label_list = []
        batch_str = []
        id_list = []
        original_len = []

        for sam in samples:
            try:
                if self.task == 'intent':
                    label = sam['intent']
                    label = self.label2idx(label)
                    label_list.append(label)
                elif self.task == 'slot':
                    label_seq = [self.label2idx(i) for i in sam['tags']] + \
                                [self.label2idx('O') for _ in range(self.max_len - len(sam['tags']))]
                    label_list.append(label_seq)

            except:
                pass
                #print('Error in collate_fn')

            if self.task == 'intent':
                batch_str.append(sam['text'].split())
            elif self.task == 'slot':
                batch_str.append(sam['tokens'])
                original_len.append(len(sam['tokens']))

            id_list.append(sam['id'])

        encoding_list = self.vocab.encode_batch(batch_str, to_len=self.max_len)
        encoding_list = torch.LongTensor(encoding_list)
        label_list = torch.LongTensor(label_list)

        return {'tensor': encoding_list, 'label': label_list, 'id': id_list, 'original_len': original_len}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
