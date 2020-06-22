import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from new_model import AttentionNetwork
from dataset import TweetsDataset
from utils import *
import json
import click
import os
import copy


def main(config, save_checkpoint_path, seed=None):
    """
    Training and validation.
    """
    global checkpoint, start_epoch
    # get configs
    config_dict = get_config(config)
    config = config_to_namedtuple(config_dict)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(config)
    # Model parameters
    n_classes = config.model.n_classes
    word_rnn_size = config.model.word_rnn_size  # word RNN size
    word_rnn_layers = config.model.word_rnn_layers  # number of layers in character RNN
    word_att_size = config.model.word_att_size  # size of the word-level attention layer (also the size of the word context vector)
    dropout = config.model.dropout  # dropout
    fine_tune_word_embeddings = config.model.fine_tune_word_embeddings  # fine-tune word embeddings?
    sentence_length_cut = config.model.sentence_length_cut #set fixed sentence length

    # Training parameters
    start_epoch = config.training.start_epoch  # start at this epoch
    batch_size = config.training.batch_size  # batch size
    lr = config.training.lr  # learning rate
    momentum = config.training.momentum  # momentum
    workers = config.training.workers  # number of workers for loading data in the DataLoader
    epochs = config.training.epochs  # number of epochs to run
    checkpoint = config.training.checkpoint  # path to saved model checkpoint, None if none

    cudnn.benchmark = False  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model or load checkpoint
    if checkpoint!="None":
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        # word_map = checkpoint['word_map']
        start_epoch = checkpoint['epoch'] + 1
        print(
            '\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
    else:
        model = AttentionNetwork(n_classes=n_classes,
                                 emb_sizes_list=[50],
                                 word_rnn_size=word_rnn_size,
                                 word_rnn_layers=word_rnn_layers,
                                 word_att_size=word_att_size,
                                 dropout=dropout,
                                 device=device)
        # model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(TweetsDataset("train_small_split.csv", "./dataset", sentence_length_cut = sentence_length_cut),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=device,
              config=config)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, 0.9)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, save_checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch, device, config):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()

    # Batches
    for i, data in enumerate(train_loader):

        # embeddings = torch.tensor(data["embeddings"])
        embeddings = data["embeddings"]
        labels = data["label"]

        data_time.update(time.time() - start)

        embeddings = embeddings.to(device)
        # labels = labels.squeeze(1).to(device)  # (batch_size)
        labels = labels.to(device)  # (batch_size)

        # Forward prop.
        scores, word_alphas, emb_weights = model(embeddings)

        # Loss
        loss = criterion(scores.to(device), labels)  # scalar

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients

        if config.training.grad_clip!="None":
            clip_gradient(optimizer, config.grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs.update(accuracy, labels.size(0))

        start = time.time()

        # Print training status
        if i % config.training.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  acc=accs))

@click.command()
@click.option('--config', default='configs/pipeline_check_glove.json', type=str)
@click.option('--save-checkpoint-path', default='./log_dir/')
@click.option('--seed', default=0, type=int)

def main_cli(config, save_checkpoint_path, seed):
    main(config, save_checkpoint_path, seed)


if __name__ == '__main__':
    main_cli()
