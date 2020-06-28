import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from attention_network import AttentionNetwork
# import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from dataset import TweetsDataset
from utils import *
from load_embeddings import *
from torch.utils.tensorboard import SummaryWriter
import json
import click
import os
import copy
import numpy as np


def predict(eval_loader, model, device, config, elmo):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.eval()

    # Batches
    length = config.model.sentence_length_cut
    results = np.array([])
    for i, (data, tweet) in enumerate(eval_loader):
        embeddings = data["embeddings"]
        embeddings = embeddings.to(device)
        labels = data["label"]
        elmo_embeddings = torch.Tensor(elmo.embed(np.array(tweet).T, [length for _ in range(len(labels))])).to(device)
        # print(elmo_embeddings.shape)
        embeddings = torch.cat([embeddings, elmo_embeddings], 2)

        # Forward prop.
        scores, word_alphas, emb_weights = model(embeddings)

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        results = np.concatenate((results, predictions.numpy()))

    return results


@click.command()
@click.option('--config', default='configs/pipeline_check_lstm.json', type=str)
@click.option('--save-checkpoint-path', default='./log_dir/')
@click.option('--prediction-file-path', default='./prediction', type=str)

def main_cli(config, save_checkpoint_path, prediction_file_path):
    # Dataset parameters
    config_dict = get_config(config)
    config = config_to_namedtuple(config_dict)

    dataset_path = config.dataset.dataset_dir
    train_file_path = config.dataset.rel_train_path
    test_file_path = config.dataset.rel_test_path
    sentence_length_cut = config.model.sentence_length_cut #set fixed sentence length
    workers = config.training.workers  # number of workers for loading data in the DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # setup embeddings
    glove_embedding = GloveEmbedding(dataset_path, train_file_path, test_file_path, sentence_length_cut)
    syngcn_embedding = SynGcnEmbedding(dataset_path, train_file_path, test_file_path, sentence_length_cut, "../embeddings/syngcn_embeddings.txt")
    elmo = hub.load("https://tfhub.dev/google/elmo/3")
    elmoEmbedding = ElmoEmbedding(elmo, None)

    # dataloader
    eval_loader = torch.utils.data.DataLoader(TweetsDataset(glove_embedding.get_test_set(), syngcn_embedding.get_test_set()),
                                               batch_size=100, shuffle=False,
                                               num_workers=workers, pin_memory=True)

    checkpoint = torch.load(save_checkpoint_path)
    model = checkpoint['model']

    results = predict(eval_loader, model, device, config, elmoEmbedding)
    np.savetxt(prediction_file_path, results, delimiter=',')

if __name__ == '__main__':
    main_cli()
