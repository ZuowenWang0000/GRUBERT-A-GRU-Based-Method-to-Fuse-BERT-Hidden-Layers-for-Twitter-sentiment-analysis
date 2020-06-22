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

    def __init__(self, csv_file_path, datapath):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.raw_tweets_with_labels = pd.read_csv(os.path.join(datapath, csv_file_path))
        self.embedding_lookup = get_glove_embedding(datapath)

    def __len__(self):
        return len(self.raw_tweets_with_labels)

    def __getitem__(self, idx):
        tweets = self.raw_tweets_with_labels["text"][idx]
        label = self.raw_tweets_with_labels["label"][idx]
        tweets = tokenizer(tweets)

        embeddings = np.array([self.embedding_lookup.vectors[self.embedding_lookup.stoi[x[0]]].numpy() for x in tweets])

        # return {"embeddings": torch.tensor(embeddings), "labels": torch.tensor(labels)}
        return {"embeddings": embeddings,
                "label": int(label)}

