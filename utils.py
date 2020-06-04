import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nltk import word_tokenize

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
        vocab = f_vocab.read().split()
    with open(train_file_pos_path, 'r') as f_pos:
        pos_text = f_pos.read().splitlines()
    with open(train_file_neg_path, 'r') as f_neg:
        neg_text = f_neg.read().splitlines()

    return pos_text, neg_text, vocab

def load_test():
    '''
    :parameter
    train_full:bool
        if true, load the full dataset

    :returns
    train_pos: a list of strings containing all positive texts
    train_neg: a list of strings containing all negative texts
    vocab: a list of strings containing all vocabulary

    '''
    vocab_file_path = "./dataset/vocab_cut.txt"
    with open(vocab_file_path, 'r') as f_vocab:
        vocab = f_vocab.read().split()
    with open('test_data.txt', 'r') as f_test:
        test_text = f_test.read().splitlines()
    return test_text, vocab

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

def load_embedding(vocab, embedding_path):
    """
    :parameter
    vocab: list
        a list of strings containing all vocabularies
    embedding_path: string
        the path to the embedding, i.e. std_glove_embeddings
    :return
    embeddings_dict: dict
        a dictionary contains mapping from words to vectors
    """
    embeddings_dict = {}
    with np.load(embedding_path, 'r') as f:
        """
            f['arr_0'] is word embedding, f['arr_1'] is context embedding
        """
        embeddings_dict = dict(zip(vocab, f['arr_0']))
    return embeddings_dict

def preprocess(text, stop_words):
    text = text.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc