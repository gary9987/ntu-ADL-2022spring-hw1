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

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    checkpoint_path = './checkpoint/intent/'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)


    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # crecate DataLoader for train / dev datasets
    trainloader = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=2
                                              , collate_fn=datasets['train'].collate_fn)
    validloader = torch.utils.data.DataLoader(datasets['eval'], batch_size=args.batch_size, shuffle=False, num_workers=2
                                              , collate_fn=datasets['eval'].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          num_class=datasets['train'].num_classes,
                          max_len=args.max_len)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.to(device)

    # init optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(5, 100, 10)], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

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
        net.train()

        for package in trainloader:
            # move tensors to GPU if CUDA is available
            data = package['tensor']
            target = package['label']
            data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)['outputs']

            # select the class with highest probability
            _, pred = output.max(1)

            # if the model predicts the same results as the true
            # label, then the correct counter will plus 1
            train_correct += pred.eq(target).sum().item()

            # calculate the batch loss
            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        net.eval()
        for package in validloader:

            data = package['tensor']
            target = package['label']
            # move tensors to GPU if CUDA is available
            data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)['outputs']

            # select the class with highest probability
            _, pred = output.max(1)

            # if the model predicts the same results as the true
            # label, then the correct counter will plus 1
            valid_correct += pred.eq(target).sum().item()

            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(trainloader.dataset.data)
        valid_loss = valid_loss / len(validloader.dataset.data)
        train_correct = 100. * train_correct / len(trainloader.dataset.data)
        valid_correct = 100. * valid_correct / len(validloader.dataset.data)

        # print training/validation statistics
        print(
            '\tTraining Acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Acc: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_correct,
                train_loss, valid_correct, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(net.state_dict(), str(args.ckpt_dir) + '/best.pt')
            valid_loss_min = valid_loss

        scheduler.step()

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=56)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
