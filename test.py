import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def test(eval_loader, model, criterion, optimizer, epoch, device, config, tf_writer, prepare_embeddings_fn, embedder):
    """
    Performs one epoch's validation.

    :param eval_loader: DataLoader for validation data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.eval()  # eval mode disables dropout

    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    # Batches
    for _, data in enumerate(eval_loader):

        # Perform embedding + padding
        embeddings, labels = prepare_embeddings_fn(data, embedder, device)

        # Forward prop.
        output = model(embeddings)

        if config.model.use_regularization == "none":
            loss = criterion(output["logits"].to(device), labels)
        elif config.model.use_regularization == "l1":
            # Regularization on embedding weights
            emb_weights_norm = torch.norm(model.emb_weights, p=1)
            # Loss
            loss = criterion(output["logits"].to(device), labels) + config.model.regularization_lambda * emb_weights_norm  # scalar
        else:
            raise NotImplementedError("Regularization other than 'none' or 'l1' not supported")

        # Find accuracy
        _, predictions = output["logits"].max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        accs.update(accuracy, labels.size(0))
    
        try:
            for sentence in data:
                sentence.clear_embeddings()
        except:
            pass

    # Print eval status
    print('Evaluation:\t'
                  'Eval Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Eval Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(loss=losses, acc=accs), flush=True)

    # ...log the running loss, accuracy
    tf_writer.add_scalar('test loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('test accuracy (avg. epoch)', accs.avg, epoch)
