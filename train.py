import sys
import os
import random
import torch
import numpy as np
if __name__ == "__main__":
    try:
        # Try to set the random seed, have to do this here instead of in main()
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        print("Using seed: %d" % seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        seed = None
        print("WARNING: Seed not set")

import time
import sys
import copy
import json
import click

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from lstm_model import LstmModel
from bert_model import BertMixModel, BertBaseModel, BertWSModel, BertSentimentModel, BertLastFourModel, BertMixLinearModel, \
    BertMixLSTMModel, RobertaSentimentModel, RobertaWSModel
from flair_model import GSFlairMixModel
from dataset import BertTwitterDataset, RobertaTwitterDataset
from utils import *
from test import *
from embeddings import *

def main(config, seed=None, embedding="bert-mix"):
    """
    Training and validation.
    """
    # Get config
    config_dict = get_config(config)
    config = config_to_namedtuple(config_dict)

    print(config)
    model_type = eval(config.model.architecture)  # Find out which type of model to instantiate

    n_classes = config.model.n_classes
    # fine_tune_embeddings = config.model.fine_tune_embeddings  # fine-tune word embeddings?
    sentence_length_cut = config.model.sentence_length_cut #set fixed sentence length

    # Training parameters
    start_epoch = config.training.start_epoch  # start at this epoch
    batch_size = config.training.batch_size  # batch size
    lr = config.training.lr  # learning rate
    momentum = config.training.momentum  # momentum
    workers = config.training.workers  # number of workers for loading data in the DataLoader
    epochs = config.training.epochs  # number of epochs to run
    checkpoint = config.training.checkpoint  # path to saved model checkpoint, None if none
    save_checkpoint_freq_epoch = config.training.save_checkpoint_freq_epoch
    train_without_val = config.training.train_without_val
    # Replace __USER__ with actual username, append seed for uniqueness
    save_checkpoint_path = config.training.save_checkpoint_path.replace("__USER__", os.popen("whoami").read().strip()) + f"_seed{seed}"
    weight_decay = config.training.weight_decay  # weight decay
    lr_decay = config.training.lr_decay  # learning rate decay

    # Dataset parameters
    dataset_path = config.dataset.dataset_dir
    train_file_path = config.dataset.rel_train_path
    val_file_path = config.dataset.rel_val_path
    test_file_path = config.dataset.rel_test_path

    setattr(config.model, "embedding_type", embedding)  # Add embedding type to model config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(config.model, "device", device)  # Add device to model config

    print("Checkpoints will be saved in: %s" % save_checkpoint_path, flush=True)

    print(f"[{embedding}] initializing embedder", flush=True)

    if embedding in ["gs-flair", "flair", "gs-bert", "gs-elmo", "elmo", "glove", "syngcn", "gs-only", "twitter"]:
        import flair
        from flair.datasets import CSVClassificationDataset
        print(f"[{embedding}] initializing dataset", flush=True)

        # Initialize datasets
        train_dataset = CSVClassificationDataset(os.path.join(dataset_path, train_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        val_dataset = CSVClassificationDataset(os.path.join(dataset_path, val_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        
        # Initialize data loaders
        train_loader = flair.datasets.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = flair.datasets.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

        # Tell training loop how to prepare embeddings
        prepare_embeddings_fn = prepare_embeddings_flair
        print(f"[{embedding}] entering training loop", flush=True)
    
    elif embedding in ["bert-base", "bert-mix", "bert-last-four", "roberta-mix"]:
        print("[" + embedding + "]" + " initializing embeddings+dataset", flush=True)

        # Initialize datasets
        if embedding == "roberta-mix":
            train_dataset = RobertaTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path),
                                               sentence_length_cut=sentence_length_cut)
            val_dataset = RobertaTwitterDataset(csv_file=os.path.join(dataset_path, val_file_path),
                                             sentence_length_cut=sentence_length_cut)
        else:  # using bert class embedding
            train_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path), sentence_length_cut=sentence_length_cut)
            val_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, val_file_path), sentence_length_cut=sentence_length_cut)

        # Initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)  # should shuffle really be false? copying from the notebook
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)

        # Tell training loop how to prepare embeddings
        prepare_embeddings_fn = eval("prepare_embeddings_" + embedding.replace("-", "_"))
        print("[" + embedding + "]" + " entering training loop", flush=True)

    else:
        raise NotImplementedError("Unsupported embedding: " + embedding)

    # set up tensorboard writer
    writer = SummaryWriter(save_checkpoint_path)

    # Initialize model or load checkpoint
    if checkpoint != "none":
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1), flush=True)
    else:
        model = model_type(n_classes=n_classes, model_config=config.model)
        print("Instantiated new model", flush=True)
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    if "elmo" in embedding:  # can't save elmo embedder somehow, so have to use it outside the model
        print("Using elmo, overriding model embedder", flush=True)
        model.embedder = None
        embedder = initialize_embeddings("elmo", device, fine_tune_embeddings=False)
    elif hasattr(model, "embedder"):
        print("Model has built-in embedder, using it", flush=True)
        embedder = model.embedder  # Use embedder inside the model, this allows saving it (e.g. in case it is fine-tuned)
    else:
        # Use embedder from outside the model
        print("Using user-defined embedder", flush=True)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # Epochs
    train_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=device,
              config=config,
              tf_writer=writer,
              prepare_embeddings_fn=prepare_embeddings_fn,
              embedder=embedder)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, lr_decay)

        # Save checkpoint and perform validation
        if epoch % save_checkpoint_freq_epoch == 0:
            save_checkpoint(epoch, model, optimizer, save_checkpoint_path)
            if not train_without_val:
                test(val_loader, model, criterion, epoch, device, config, writer, prepare_embeddings_fn, embedder)
        epoch_end = time.time()
        print("Per epoch time = {}".format(epoch_end-epoch_start), flush=True)

    train_end_time = time.time()
    print("Total training time: {} minutes".format((train_end_time-train_start_time)/60.0), flush=True)

    print("Final evaluation:", flush=True)
    test(val_loader, model, criterion, epoch, device, config, writer, prepare_embeddings_fn, embedder)
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, prepare_embeddings_fn, embedder):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param device: device on which to perform training
    :param config: config dict read in from JSON
    :param tf_writer: TensorBoard writer for logging
    :param prepare_embeddings_fn: function to perform embedding that should take the following arguments (1) data as returned by data loader,
        (2) embedder to use for performing embedding, (3) device on which to perform embedding, (4) params (config is passed here)
    :param embedder: embedder to use for embedding, passed to prepare_embeddings_fn
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()
    # Batches
    for i, data in enumerate(train_loader):

        # Perform embedding + padding if necessary
        embeddings, labels = prepare_embeddings_fn(data, embedder, device, config)
        data_time.update(time.time() - start)

        # Forward prop.
        output = model(embeddings)

        # Regularization on embedding weights -- not all models support this
        if config.model.use_regularization == "none":
            loss = criterion(output["logits"].to(device), labels)
        elif config.model.use_regularization == "l1":
            # Regularization on embedding weights
            emb_weights_norm = torch.norm(model.emb_weights, p=1)
            # Loss
            loss = criterion(output["logits"].to(device), labels) + config.model.regularization_lambda * emb_weights_norm  # scalar
        else:
            raise NotImplementedError("Regularization other than 'none' or 'l1' not supported")

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if config.training.grad_clip != "none":
            clip_gradient(optimizer, config.grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        _, predictions = output["logits"].max(dim=1)  # (n_documents)
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
                                                                  acc=accs), flush=True)

        # Delete embeddings from flair, somehow it runs out of memory otherwise
        try:
            for sentence in data:
                sentence.clear_embeddings()
        except:
            pass

    # Log the running loss, accuracy
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)


@click.command()
@click.option('-c', '--config', required=True, type=str, help="JSON file specifying model and training parameters")
@click.option('-s', '--seed', default=0, type=int, help="Random seed to use for producing reproducible results")
@click.option('-e', '--embedding', required=True, type=str, help="Embedding to use for training")

def main_cli(config, seed, embedding):
    main(config, seed, embedding)


if __name__ == '__main__':
    main_cli()
