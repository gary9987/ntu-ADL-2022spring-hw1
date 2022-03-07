import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from tqdm import trange, tqdm

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier
from sklearn.model_selection import train_test_split

from nni.utils import merge_parameter
import nni
import logging

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
torch.manual_seed(0)
logger = logging.getLogger('train_intent')


def main(args):

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len, task='slot')
        for split, split_data in data.items()
    }
    # crecate DataLoader for train / dev datasets
    trainloader = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4
                                              , collate_fn=datasets['train'].collate_fn)
    validloader = torch.utils.data.DataLoader(datasets['eval'], batch_size=args.batch_size, shuffle=False, num_workers=4
                                              , collate_fn=datasets['eval'].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          num_class=datasets['train'].num_classes,
                          task='slot')

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    device = args.device
    net = model.to(device)

    valid_loss_min = np.Inf
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # keep track of training and validation loss
        train_correct = 0
        valid_correct = 0
        train_loss = 0.0
        valid_loss = 0.0

        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        net.eval()

        for package in trainloader:
            # move tensors to GPU if CUDA is available
            data = package['tensor']
            target = package['label']
            data, target = data.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)['outputs'].permute(0, 2, 1)

            # select the class with highest probability
            _, pred = output.max(1)

            # if the model predicts the same results as the true
            # label, then the correct counter will plus
            print(pred[0])
            print(target[0])
            train_correct += torch.all(pred.eq(target), dim=1).sum().item()

        print(train_correct / len(trainloader.dataset))



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
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
        help="Directory to save the model file.",
        default="./ckpt/slot/best.pt",
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1.0)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = merge_parameter(parse_args(), tuner_params)
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise

    '''
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
    '''