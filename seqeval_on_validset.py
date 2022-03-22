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
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def joint_accuracy(pred: list, label: list):
    correct = sum(list(map(lambda x, y: x == y, pred, label)))
    return correct / len(pred)


def token_accuracy(pred: list, label: list):
    correct = sum(list(map(lambda x, y: sum(list(map(lambda i, j: i == j, x, y))), pred, label)))
    length = sum(list(map(lambda x: len(x), pred)))
    return correct / length


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
    label_list = []

    # Predict dataset
    for package in dataloader:
        # move tensors to GPU if CUDA is available
        data = package['tensor'].to(device)
        labels = package['label']

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)['outputs'].permute(0, 2, 1)

        # select the class with highest probability
        _, pred = output.max(1)
        for i in range(len(pred)):
            tmp = [dataset.idx2label(tag.item()) for tag in pred[i]]
            pred_list.append(tmp)

        for i in range(len(labels)):
            tmp = [dataset.idx2label(tag.item()) for tag in labels[i]]
            label_list.append(tmp)

    print('Joint Accuracy: ', joint_accuracy(pred_list, label_list))
    print('Token Accuracy: ', token_accuracy(pred_list, label_list))
    print(classification_report(label_list, pred_list, mode='strict', scheme=IOB2))



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default='./data/slot/eval.json'
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
        default='./ckpt/slot/best.pt'
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot_tag.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

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
