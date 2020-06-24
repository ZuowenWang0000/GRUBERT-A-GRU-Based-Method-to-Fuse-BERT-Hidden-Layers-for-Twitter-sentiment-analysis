import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from attention_network import AttentionNetwork
from dataset import TweetsDataset
from utils import *
import json
import click
import os
import copy


def test(eval_loader, model, criterion, device, config):
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
    for i, data in enumerate(eval_loader):
        embeddings = data["embeddings"]
        labels = data["label"]
        embeddings = embeddings.to(device)
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
