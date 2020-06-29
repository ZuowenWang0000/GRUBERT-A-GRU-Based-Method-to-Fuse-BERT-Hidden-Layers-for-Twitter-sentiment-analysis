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

def test_flair(eval_loader, model, criterion, device, config, tf_writer, epoch, embedder):
    """
    Performs one epoch's testing.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.eval()  # training mode enables dropout
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    # Batches
    length = config.model.sentence_length_cut
    for i, sentences in enumerate(eval_loader):
        # batch_start = time.time()
        # embeddings = torch.tensor(data["embeddings"])

        # Perform embedding + padding
        embedder.embed(sentences)

        lengths = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            embedder.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : embedder.embedding_length * nb_padding_tokens
                ]
                all_embs.append(t)

        embeddings = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                embedder.embedding_length,
            ]
        )

        embeddings = embeddings.to(device)
        labels = torch.as_tensor(np.array([int(s.labels[0].value) for s in sentences]))
        # print(labels)
        # print(labels.shape)
        # print(embeddings.shape)

        # print(elmo_embeddings.shape)
        labels = labels.to(device)  # (batch_size)

        scores, word_alphas, emb_weights = model(embeddings)

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
                  'Eval Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(loss=losses, acc=accs), flush=True)

    # ...log the running loss, accuracy
    tf_writer.add_scalar('test loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('test accuracy (avg. epoch)', accs.avg, epoch)

def test(eval_loader, model, criterion, device, config, tf_writer, epoch, embedder):
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
        elmo_embeddings = torch.Tensor(embedder.embed(np.array(tweet).T, [length for _ in range(len(labels))])).to(device)
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
                  'Eval Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(loss=losses, acc=accs), flush=True)

    # ...log the running loss, accuracy
    tf_writer.add_scalar('test loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('test accuracy (avg. epoch)', accs.avg, epoch)
