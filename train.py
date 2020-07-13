import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from attention_network import AttentionNetwork
from lstm_model import LstmModel
from gru_model import GruModel
from bert_model import BertSentimentModel, BertSentimentModelMix2, BertSentimentModelMixSix
from dataset import BertTwitterDataset
from utils import *
import json
import click
import os
import copy
from test import *
# from load_embeddings import *
from torch.utils.tensorboard import SummaryWriter
# import tensorflow_hub as hub
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# import tensorflow as tf
# tf.disable_eager_execution()
import sys
import numpy as np

def main(config, save_checkpoint_path, seed=None, embedding="elmo", fine_tune=False):
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
    # embedding parameters , the center of our study
    # emb_sizes_list = config.embeddings.emb_sizes_list

    # Model parameters
    if config.model.architecture == "attention":
        print("using attention model")
        model_type = AttentionNetwork
    elif config.model.architecture == "lstm":
        print("using lstm model")
        model_type = LstmModel
    elif config.model.architecture == "gru":
        print("using gru architecture")
        model_type = GruModel
    elif config.model.architecture == "bert":
        print("using bert architecture")
        model_type = BertSentimentModel
    elif config.model.architecture == "bert-mix2":
        print("using two groups for bert")
        model_type = BertSentimentModelMix2
    elif config.model.architecture == "bert-mix6":
        print("using 6 groups of 2")
        model_type = BertSentimentModelMixSix
    else:
        raise NotImplementedError
    
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
    save_checkpoint_freq_epoch = config.training.save_checkpoint_freq_epoch
    train_without_val = config.training.train_without_val
    save_checkpoint_path = config.training.save_checkpoint_path

    # Dataset parameters
    dataset_path = config.dataset.dataset_dir
    train_file_path = config.dataset.rel_train_path
    val_file_path = config.dataset.rel_val_path
    test_file_path = config.dataset.rel_test_path

    cudnn.benchmark = False  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # elmo = hub.Module("https://tfhub.dev/google/elmo/3")
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # elmoEmbedding = ElmoEmbedding(elmo, sess)
    # embedding = elmoEmbedding.embed(words_to_embed, sess)
    # embedding = sess.run(embedding_tensor)
    # print(embedding.shape)


    if embedding == "flair":
        from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
        print("[flair] initializing embeddings", flush=True)
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        # elmo_embedding = ELMoEmbeddings(model="medium", embedding_mode="average")
        flair_forward_embedding = FlairEmbeddings("mix-forward", chars_per_chunk=64, fine_tune=fine_tune)
        flair_backward_embedding = FlairEmbeddings("mix-backward", chars_per_chunk=64, fine_tune=fine_tune)
        # embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, elmo_embedding])
        embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, flair_forward_embedding, flair_backward_embedding])

        import flair
        from flair.datasets import CSVClassificationDataset
        print("[flair] initializing datasets", flush=True)
        train_dataset = CSVClassificationDataset(os.path.join(dataset_path, train_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        val_dataset = CSVClassificationDataset(os.path.join(dataset_path, val_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        train_loader = flair.datasets.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = flair.datasets.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        embedder = embedding.to(device)

        train_function = train_flair
        test_function = test_flair
        print("[flair] entering training loop", flush=True)

    elif embedding == "bert":
        from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
        print("[flair] initializing embeddings", flush=True)
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        # elmo_embedding = ELMoEmbeddings(model="medium", embedding_mode="average")
        bert_embedding = TransformerWordEmbeddings('bert-base-uncased', layers='-1', fine_tune=fine_tune)
        # embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, elmo_embedding])
        embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, bert_embedding])

        import flair
        from flair.datasets import CSVClassificationDataset
        print("[bert] initializing datasets", flush=True)
        train_dataset = CSVClassificationDataset(os.path.join(dataset_path, train_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        val_dataset = CSVClassificationDataset(os.path.join(dataset_path, val_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        train_loader = flair.datasets.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = flair.datasets.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        embedder = embedding.to(device)

        train_function = train_flair
        test_function = test_flair
        print("[bert] entering training loop", flush=True)
    
    elif embedding == "bert-mix":
        print("[bert-mix] initializing embeddings+dataset", flush=True)
        train_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path))
        val_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, val_file_path))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)  # should shuffle really be false? copying from the notebook
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        embedder = None

        train_function = train_bert_mix
        test_function = test_bert_mix
        print("[bert-mix] entering training loop", flush=True)
    elif embedding == "bert-mix2":
        print("[bert-mix-two] initializing embeddings+dataset", flush=True)
        train_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path))
        val_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, val_file_path))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
                                                   shuffle=False)  # should shuffle really be false? copying from the notebook
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        embedder = None

        train_function = train_bert_mix_two
        test_function = test_bert_mix_two
        print("[bert-mix] entering training loop", flush=True)
    elif embedding == "bert-mix6":
        print("[bert-mix-six] initializing embeddings+dataset", flush=True)
        train_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path))
        val_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, val_file_path))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
                                                   shuffle=False)  # should shuffle really be false? copying from the notebook
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        embedder = None

        train_function = train_bert_mix_six
        test_function = test_bert_mix_six
        print("[bert-mix-six] entering training loop", flush=True)


    else:
        from flair.embeddings import WordEmbeddings, ELMoEmbeddings, StackedEmbeddings
        print("[elmo] initializing embeddings", flush=True)
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        # elmo_embedding = ELMoEmbeddings(model="medium", embedding_mode="average")
        elmo_embedding = ELMoEmbeddings(model="medium", embedding_mode="top")
        # embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, elmo_embedding])
        embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, elmo_embedding])

        import flair
        from flair.datasets import CSVClassificationDataset
        print("[elmo] initializing datasets", flush=True)
        train_dataset = CSVClassificationDataset(os.path.join(dataset_path, train_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        val_dataset = CSVClassificationDataset(os.path.join(dataset_path, val_file_path), {0: "text", 1: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        train_loader = flair.datasets.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = flair.datasets.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        embedder = embedding.to(device)

        train_function = train_flair
        test_function = test_flair
        print("[elmo] entering training loop", flush=True)
        # import tensorflow as tf
        # import tensorflow_hub as hub
        # glove_embedding = GloveEmbedding(dataset_path, train_file_path, val_file_path, sentence_length_cut)
        # syngcn_embedding = SynGcnEmbedding(dataset_path, train_file_path, val_file_path, sentence_length_cut, "../embeddings/syngcn_embeddings.txt")


        # # DataLoaders
        # train_loader = torch.utils.data.DataLoader(TweetsDataset(glove_embedding.get_train_set(), syngcn_embedding.get_train_set()),
        #                                         batch_size=batch_size, shuffle=True,
        #                                         num_workers=workers, pin_memory=True)
        # #    validation
        # val_loader = torch.utils.data.DataLoader(TweetsDataset(glove_embedding.get_test_set(), syngcn_embedding.get_test_set()),
        #                                         batch_size=batch_size, shuffle=True,
        #                                         num_workers=workers, pin_memory=True)

        # # initialzie elmo
        # # sess = tf.Session()
        # elmo = hub.load("https://tfhub.dev/google/elmo/3")
        # # sess.run(tf.global_variables_initializer())
        # embedder = ElmoEmbedding(elmo, None)
        # train_function = train
        # test_function = test

    # writer.add_graph(model)

    # set up tensorboard writer
    writer = SummaryWriter(save_checkpoint_path)

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
        temp_list = []#[e.embedding_length for e in embedding.embeddings] if (embedding != "bert-mix") or (embedding != "bert-mix2") else []
        model = model_type(n_classes=n_classes,
                                 emb_sizes_list=temp_list,
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

    # Epochs
    train_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        # One epoch's training
        train_function(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=device,
              config=config,
              tf_writer=writer,
              embedder=embedder)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, 0.99)

        # Save checkpoint
        if epoch % save_checkpoint_freq_epoch == 0:
            save_checkpoint(epoch, model, optimizer, save_checkpoint_path)
            if not train_without_val:
                test_function(val_loader, model, criterion, device, config, writer, epoch, embedder)
        epoch_end = time.time()
        print("per epoch time = {}".format(epoch_end-epoch_start))
        sys.stdout.flush()

    train_end_time = time.time()
    print("Total training time : {} minutes".format((train_end_time-train_start_time)/60.0))

    print("Final evaluation:")
    test_function(val_loader, model, criterion, device, config, writer, epoch, embedder)
    writer.close()


def train_flair(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, embedder):
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
    length = config.model.sentence_length_cut
    for i, sentences in enumerate(train_loader):
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
        # batch_load = time.time()
        data_time.update(time.time() - start)


        # print(embeddings.shape)

        # print("batch load time:{}".format(batch_load - batch_start))
        # Forward prop.
        scores, word_alphas, emb_weights = model(embeddings)

        if config.embeddings.use_regularization == "none":
            loss = criterion(scores.to(device), labels)
        elif config.embeddings.use_regularization == "l1":
            # Regularization on embedding weights
            emb_weights_norm = torch.norm(model.emb_weights, p=1)
            # Loss
            loss = criterion(scores.to(device), labels) + config.embeddings.l1_lambda * emb_weights_norm  # scalar
        else:
            raise NotImplementedError

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # print(model.emb_weights.grad)

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
                                                                  acc=accs), flush=True)
        batch_end = time.time()
        for sentence in sentences:
            sentence.clear_embeddings()
        # print("batch time :{}".format(batch_end - batch_start))
    # ...log the running loss, accuracy
    print("***writing to tf board")
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
def train_bert_mix_six(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, embedder):
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
    length = config.model.sentence_length_cut
    for i, data in enumerate(train_loader):
        # batch_start = time.time()
        # embeddings = torch.tensor(data["embeddings"])
        x = data["text"]
        labels = data["label"]
        embeddings = model.model(input_ids=x.to(device))
        labels = labels.to(device)

        #h0 = torch.cat(embeddings[2][1:5], 2)
        #h1 = torch.cat(embeddings[2][5:9], 2)
        #h2 = torch.cat(embeddings[2][9:13], 2)

        h0 = torch.cat(embeddings[2][1:3],2)
        h1 = torch.cat(embeddings[2][3:5],2)
        h2 = torch.cat(embeddings[2][5:7],2)
        h3 = torch.cat(embeddings[2][7:9],2)
        h4 = torch.cat(embeddings[2][9:11],2)
        h5 = torch.cat(embeddings[2][11:13],2)


        data_time.update(time.time() - start)

        # Forward prop.
        #scores, word_alphas, emb_weights = model([h0, h1, h2])
        scores, word_alphas, emb_weights = model([h0, h1, h2, h3, h4, h5])

        if config.embeddings.use_regularization == "none":
            loss = criterion(scores.to(device), labels)
        else:
            raise NotImplementedError

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if config.training.grad_clip != "None":
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
                                                                  acc=accs), flush=True)
        batch_end = time.time()
        # print("batch time :{}".format(batch_end - batch_start))
    # ...log the running loss, accuracy
    print("***writing to tf board")
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
def train_bert_mix_two(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, embedder):
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
    length = config.model.sentence_length_cut
    for i, data in enumerate(train_loader):
        # batch_start = time.time()
        # embeddings = torch.tensor(data["embeddings"])
        x = data["text"]
        labels = data["label"]
        embeddings = model.model(input_ids=x.to(device))
        labels = labels.to(device)

        #h0 = torch.cat(embeddings[2][1:5], 2)
        #h1 = torch.cat(embeddings[2][5:9], 2)
        #h2 = torch.cat(embeddings[2][9:13], 2)

        h0 = torch.cat(embeddings[2][1:7],2)
        h1 = torch.cat(embeddings[2][7:13],2)


        data_time.update(time.time() - start)

        # Forward prop.
        #scores, word_alphas, emb_weights = model([h0, h1, h2])
        scores, word_alphas, emb_weights = model([h0, h1])

        if config.embeddings.use_regularization == "none":
            loss = criterion(scores.to(device), labels)
        else:
            raise NotImplementedError

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if config.training.grad_clip != "None":
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
                                                                  acc=accs), flush=True)
        batch_end = time.time()
        # print("batch time :{}".format(batch_end - batch_start))
    # ...log the running loss, accuracy
    print("***writing to tf board")
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

def train_bert_mix(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, embedder):
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
    length = config.model.sentence_length_cut
    for i, data in enumerate(train_loader):
        # batch_start = time.time()
        # embeddings = torch.tensor(data["embeddings"])
        x = data["text"]
        labels = data["label"]
        embeddings = model.model(input_ids=x.to(device))
        labels = labels.to(device)

        h0 = torch.cat(embeddings[2][1:5], 2)
        h1 = torch.cat(embeddings[2][5:9], 2)
        h2 = torch.cat(embeddings[2][9:13], 2)

        data_time.update(time.time() - start)

        # Forward prop.
        scores, word_alphas, emb_weights = model([h0, h1, h2])

        if config.embeddings.use_regularization == "none":
            loss = criterion(scores.to(device), labels)
        else:
            raise NotImplementedError

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if config.training.grad_clip != "None":
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
                                                                  acc=accs), flush=True)
        batch_end = time.time()
        # print("batch time :{}".format(batch_end - batch_start))
    # ...log the running loss, accuracy
    print("***writing to tf board")
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)



def train(train_loader, model, criterion, optimizer, epoch, device, config, tf_writer, embedder):
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
    length = config.model.sentence_length_cut
    for i, (data, tweet) in enumerate(train_loader):
        # batch_start = time.time()
        # embeddings = torch.tensor(data["embeddings"])
        embeddings = data["embeddings"]
        labels = data["label"]
        embeddings = embeddings.to(device)


        elmo_embeddings = torch.Tensor(embedder.embed(np.array(tweet).T, [length for _ in range(len(labels))])).to(device)
        # print(elmo_embeddings.shape)
        labels = labels.to(device)  # (batch_size)

        embeddings = torch.cat([embeddings, elmo_embeddings], 2)
        # batch_load = time.time()
        data_time.update(time.time() - start)


        # print(embeddings.shape)

        # print("batch load time:{}".format(batch_load - batch_start))
        # Forward prop.
        scores, word_alphas, emb_weights = model(embeddings)

        if config.embeddings.use_regularization == "none":
            loss = criterion(scores.to(device), labels)
        elif config.embeddings.use_regularization == "l1":
            # Regularization on embedding weights
            emb_weights_norm = torch.norm(model.emb_weights, p=1)
            # Loss
            loss = criterion(scores.to(device), labels) + config.embeddings.l1_lambda * emb_weights_norm  # scalar
        else:
            raise NotImplementedError

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # print(model.emb_weights.grad)

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
                                                                  acc=accs), flush=True)
        batch_end = time.time()
        # print("batch time :{}".format(batch_end - batch_start))
    # ...log the running loss, accuracy
    print("***writing to tf board")
    tf_writer.add_scalar('training loss (avg. epoch)', losses.avg, epoch)
    tf_writer.add_scalar('training accuracy (avg. epoch)', accs.avg, epoch)
    tf_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)



@click.command()
@click.option('--config', default='configs/pipeline_check_lstm.json', type=str)
@click.option('--save-checkpoint-path', default='./log_dir/')
@click.option('--seed', default=0, type=int)
@click.option('--embedding', default='elmo', type=str)
# @click.option('--use-flair', default=False, type=bool)
# @click.option('--use-bert', default=False,type=bool)
@click.option('--fine-tune', default=False, type=bool)

def main_cli(config, save_checkpoint_path, seed, embedding, fine_tune):
    main(config, save_checkpoint_path, seed, embedding, fine_tune)


if __name__ == '__main__':
    main_cli()
