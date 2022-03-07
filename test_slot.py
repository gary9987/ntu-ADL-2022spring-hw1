import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.utils.data

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len, task='slot')

    # Crecate DataLoader for test dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                             collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        task='slot'
    )
    model.eval()

    device = args.device
    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(device)

    pred_list = []
    id_list = []
    # Predict dataset
    for package in dataloader:
        # move tensors to GPU if CUDA is available
        data = package['tensor'].to(device)
        id_ = package['id']
        len_list = package['original_len']
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)['outputs'].permute(0, 2, 1)

        # select the class with highest probability
        _, pred = output.max(1)
        for i in range(len(pred)):
            tmp = [tag.item() for tag in pred[i]]
            tmp = tmp[:len_list[i]]
            pred_list.append(tmp)
        id_list += [i for i in id_]

    # Write prediction to file (args.pred_file)
    print(args.pred_file)
    with open(args.pred_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'tags'])
        for i in range(len(pred_list)):
            label = ''
            for t in pred_list[i]:
                label += dataset.idx2label(t) + ' '
            # Remove last space
            label = label[:-1]
            writer.writerow([id_list[i], label])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot_tag.csv")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
