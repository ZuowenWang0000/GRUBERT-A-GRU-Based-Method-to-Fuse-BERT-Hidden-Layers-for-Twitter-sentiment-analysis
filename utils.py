import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def load_train(train_on_full = False):
    '''
    :parameter
    train_full:bool
        if true, load the full dataset

    :returns
    train_pos: a list of strings containing all positive texts
    train_neg: a list of strings containing all negative texts
    vocab: a list of strings containing all vocabulary

    '''
    if(train_on_full == False):
        train_file_pos_path = "./dataset/train_pos.txt"
        train_file_neg_path = "./dataset/train_neg.txt"
        vocab_file_path = "./dataset/vocab_cut.txt"
    else:
        train_file_pos_path = "./dataset/train_pos_full.txt"
        train_file_neg_path = "./dataset/train_neg_full.txt"
        vocab_file_path = "./dataset/vocab_cut.txt"

    with open(vocab_file_path, 'r') as f_vocab:
        vocab = f_vocab.readlines()
    with open(train_file_pos_path, 'r') as f_pos:
        pos_text = f_pos.readlines()
    with open(train_file_neg_path, 'r') as f_neg:
        neg_text = f_neg.readlines()

    return pos_text, neg_text, vocab

def make_labels(size_pos, size_neg):
    '''
    :parameter
    size_pos: int
        number of tweets of positive sentiment
    size_neg: int
        number of tweets of negative sentiment

    :returns
    a list of labels +1, -1 of [size_pos times of 1, size_neg times of -1]

    '''

    return [1 for i in range(size_pos)] + [-1 for i in range(size_neg)]