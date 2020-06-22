from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from load_embeddings import get_glove_embedding

def tokenizer(x):

    return x.split()


class TweetsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_path, datapath, sentence_length_cut = 40):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.raw_tweets_with_labels = pd.read_csv(os.path.join(datapath, csv_file_path))
        self.embedding_lookup = get_glove_embedding(datapath)
        self.sentence_length_cut = sentence_length_cut

    def __len__(self):
        return len(self.raw_tweets_with_labels)

    def __getitem__(self, idx):
        tweets = self.raw_tweets_with_labels["text"][idx]
        label = self.raw_tweets_with_labels["label"][idx]

        tweets = tokenizer(tweets)


        embeddings_temp = np.array([self.embedding_lookup.vectors[self.embedding_lookup.stoi[x[0]]].numpy() for x in tweets])
        sentence_length = embeddings_temp.shape[0]
        embedding_dim = embeddings_temp.shape[1]
        if sentence_length > self.sentence_length_cut:
            embeddings = embeddings_temp[0:self.sentence_length_cut]
        else:
            embeddings = np.zeros([self.sentence_length_cut, embedding_dim])
            embeddings[0:sentence_length] = embeddings_temp

        # return {"embeddings": torch.tensor(embeddings), "labels": torch.tensor(labels)}
        return {"embeddings": embeddings,
                "label": int(label)}

