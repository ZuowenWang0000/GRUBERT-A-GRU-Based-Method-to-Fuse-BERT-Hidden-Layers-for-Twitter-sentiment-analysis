import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from utils import *


def main():
    print("running main")
    """
       load raw text, make dataset
    """
    pos_text, neg_text, vocab = load_train(train_on_full=False)
    train = pos_text + neg_text
    labels = make_labels(len(pos_text), len(neg_text))

    assert len(train) == len(labels)

    embeddings_dict = load_embedding(vocab, "./embedding/std_glove_embeddings.npz")




if __name__ == '__main__':
    main()