import time
import os
import sys
import copy
import json
import click
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from attention_network import AttentionNetwork
from lstm_model import LstmModel
from gru_model import GruModel
from bert_model import BertMixModel, BertBaseModel, BertLastFourModel
from dataset import BertTwitterDataset
from utils import *
from test import *

def main(config, seed=None, embedding="bert-mix"):
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
    model_type = eval(config.model.architecture)

    n_classes = config.model.n_classes
    fine_tune_embeddings = config.model.fine_tune_embeddings  # fine-tune word embeddings?
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
    save_checkpoint_path = config.training.save_checkpoint_path.replace("__USER__", os.popen("whoami").read().strip()) + f"_seed{seed}"
    weight_decay = config.training.weight_decay
    lr_decay = config.training.lr_decay  # 0.9 originally

    # Dataset parameters
    dataset_path = config.dataset.dataset_dir
    train_file_path = config.dataset.rel_train_path
    val_file_path = config.dataset.rel_val_path
    test_file_path = config.dataset.rel_test_path

    cudnn.benchmark = False  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Checkpoints will be saved in: %s" % save_checkpoint_path, flush=True)

    if embedding in ["flair", "bert", "elmo"]:
        import flair
        from flair.datasets import CSVClassificationDataset
        from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        embeddings_list = [glove_embedding, syngcn_embedding]

        if embedding == "flair":
            print("[flair] initializing Flair embeddings", flush=True)
            embeddings_list += [FlairEmbeddings("mix-forward", chars_per_chunk=64, fine_tune=fine_tune_embeddings), FlairEmbeddings("mix-backward", chars_per_chunk=64, fine_tune=fine_tune_embeddings)]
        elif embedding == "bert":
            print("[flair] initializing Bert embeddings", flush=True)
            embeddings_list += [TransformerWordEmbeddings('bert-base-uncased', layers='-1', fine_tune=fine_tune_embeddings)]
        elif embedding == "elmo":
            print("[flair] initializing ELMo embeddings", flush=True)
            embeddings_list += [ELMoEmbeddings(model="medium", embedding_mode="top")]
        else:
            raise NotImplementedError("Embeddings must be in ['flair', 'bert', 'elmo']")

        embedding = StackedEmbeddings(embeddings=embeddings_list)
        print("[flair] initializing dataset", flush=True)
        train_dataset = CSVClassificationDataset(os.path.join(dataset_path, train_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        val_dataset = CSVClassificationDataset(os.path.join(dataset_path, val_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        train_loader = flair.datasets.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = flair.datasets.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        embedder = embedding.to(device)
        prepare_embeddings_fn = prepare_embeddings_flair
        print("[flair] entering training loop", flush=True)
    
    elif embedding in ["bert-base", "bert-mix", "bert-last-four"]:
        print("[" + embedding + "]" + " initializing embeddings+dataset", flush=True)
        train_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path), sentence_length_cut=sentence_length_cut)
        val_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, val_file_path), sentence_length_cut=sentence_length_cut)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)  # should shuffle really be false? copying from the notebook
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        embedder = None  # embedder in model
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
        emb_sizes_list = [e.embedding_length for e in embedding.embeddings] if embedding not in ["bert-base", "bert-mix","bert-last-four"] else []
        model = model_type(n_classes=n_classes, emb_sizes_list=emb_sizes_list, model_config=config.model)
        if hasattr(model, "embedder"):
            print("Model has built-in embedder, using it", flush=True)
            embedder = model.embedder
        else:
            print("Using user-defined embedder", flush=True)

        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # Initial eval
    # print("Initial evaluation:")
    # test(eval_loader, model, criterion, optimizer, epoch, device, config, tf_writer, prepare_embeddings_fn, embedder):
    # test(val_loader, model, criterion, optimizer, 0, device, config, writer, prepare_embeddings_fn, embedder)

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

        # Save checkpoint
        if epoch % save_checkpoint_freq_epoch == 0:
            save_checkpoint(epoch, model, optimizer, save_checkpoint_path)
            if not train_without_val:
                test(val_loader, model, criterion, optimizer, epoch, device, config, writer, prepare_embeddings_fn,
                     embedder)
                # test(val_loader, model, criterion, device, config, writer, epoch, prepare_embeddings_fn, embedder)
        epoch_end = time.time()
        print("per epoch time = {}".format(epoch_end-epoch_start))
        sys.stdout.flush()

    train_end_time = time.time()
    print("Total training time: {} minutes".format((train_end_time-train_start_time)/60.0))

    print("Final evaluation:")
    test(val_loader, model, criterion, optimizer, epoch, device, config, writer, prepare_embeddings_fn, embedder)
    # test(val_loader, model, criterion, device, config, writer, epoch, embedder)
    writer.close()


def prepare_embeddings_flair(sentences, embedder, device, params):
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
        all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
        nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
        if nb_padding_tokens > 0:
            t = pre_allocated_zero_tensor[:embedder.embedding_length * nb_padding_tokens]
            all_embs.append(t)
    embeddings = torch.cat(all_embs).view([len(sentences), longest_token_sequence_in_batch, embedder.embedding_length])
    labels = torch.as_tensor(np.array([int(s.labels[0].value) for s in sentences]))
    return embeddings.to(device), labels.to(device)

def prepare_embeddings_bert_mix(data, embedder, device, params):
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))
    labels = labels.to(device)

    num_combined_per_gru = int(12 / params.model.num_grus)

    h = [torch.cat(embeddings[2][i*num_combined_per_gru+1 : (i+1)*num_combined_per_gru+1], 2) for i in range(num_grus)]
    return h, labels
    # h0 = torch.cat(embeddings[2][1:5], 2)
    # h1 = torch.cat(embeddings[2][5:9], 2)
    # h2 = torch.cat(embeddings[2][9:13], 2)
    # return [h0, h1, h2], labels

def prepare_embeddings_bert_base(data, embedder, device, params):
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))
    labels = labels.to(device)
    # h2 = torch.cat(embeddings[2][12], 2)
    h2 = embeddings[2][12]
    return [h2], labels

def prepare_embeddings_bert_last_four(data, embedder, device, params):
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))
    labels = labels.to(device)
    h2 = torch.cat(embeddings[2][9:13], 2)
    return [h2], labels


def train(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, prepare_embeddings_fn, embedder):
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

        # Perform embedding + padding
        embeddings, labels = prepare_embeddings_fn(data, embedder, device, config)
        data_time.update(time.time() - start)

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
        try:
            for sentence in data:
                sentence.clear_embeddings()
        except:
            pass

    # ...log the running loss, accuracy
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)


@click.command()
@click.option('--config', default='specify_config_using_--config_option', type=str)
@click.option('--seed', default=0, type=int)
@click.option('--embedding', default='specify_embedding_using_--embedding_option', type=str)

def main_cli(config, seed, embedding):
    main(config, seed, embedding)


if __name__ == '__main__':
    main_cli()
