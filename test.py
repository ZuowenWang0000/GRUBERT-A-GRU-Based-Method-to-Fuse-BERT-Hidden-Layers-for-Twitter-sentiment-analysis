import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from attention_network import AttentionNetwork
from dataset import TweetsDataset
from utils import *
from torch.utils.tensorboard import SummaryWriter
import json
import click
import os
import copy
import numpy as np


def test(eval_loader, model, criterion, device, config, tf_writer, epoch, elmo):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.eval()
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    # Batches
    length = config.model.sentence_length_cut
    for i, (data, tweet) in enumerate(eval_loader):
        embeddings = data["embeddings"]
        embeddings = embeddings.to(device)
        labels = data["label"]
        elmo_embeddings = torch.Tensor(elmo.embed(np.array(tweet).T, [length for _ in range(len(labels))])).to(device)
        # print(elmo_embeddings.shape)
        embeddings = torch.cat([embeddings, elmo_embeddings], 2)
        labels = labels.to(device)  # (batch_size)

        # Forward prop.
        scores, word_alphas, emb_weights = model(embeddings)

        # Loss
        loss = criterion(scores.to(device), labels)  # scalar

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        accs.update(accuracy, labels.size(0))

        # Print eval status
    print('Evaluation:\t'
                  'Eval Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Eval Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(loss=losses, acc=accs))

    # ...log the running loss, accuracy
    tf_writer.add_scalar('test loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('test accuracy (avg. epoch)', accs.avg, epoch)
