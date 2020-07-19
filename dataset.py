from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# from load_embeddings import *

def tokenizer(x):

    return x.split()


class TweetsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, glove_embedding, syngcn_embedding):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.raw_tweets_with_labels = pd.read_csv(os.path.join(datapath, csv_file_path))
        # self.embedding_lookup, self.dataset = get_glove_embedding(datapath, train_csv_file, test_csv_file, train_or_test, sentence_length_cut)
        self.glove_embedding_lookup, self.glove_dataset = glove_embedding
        self.syngcn_embedding_lookup, self.syngcn_dataset = syngcn_embedding

        # self.elmo = hub.Module("https://tfhub.dev/google/elmo/3")
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        # elmo_embedding = ElmoEmbedding(elmo, sess)
        # elmoEmbedding.embed(words_to_embed, sess)

        # self.elmo_embedding_lookup = ElmoEmbedding(self.elmo, self.sess)

    def __len__(self):
        return len(self.glove_dataset)

    def __getitem__(self, idx):
        # tweets = self.raw_tweets_with_labels["text"][idx]
        # label = self.raw_tweets_with_labels["label"][idx]

        # tweets = tokenizer(tweets)

        # embeddings_temp = np.array([self.embedding_lookup.vectors[self.embedding_lookup.stoi[x[0]]].numpy() for x in tweets])
        # sentence_length = embeddings_temp.shape[0]
        # embedding_dim = embeddings_temp.shape[1]
        # if sentence_length > self.sentence_length_cut:
        #     embeddings = embeddings_temp[0:self.sentence_length_cut]
        # else:
        #     embeddings = np.zeros([self.sentence_length_cut, embedding_dim])
        #     embeddings[0:sentence_length] = embeddings_temp
        # print([self.glove_dataset[idx].text])
        ids = self.glove_dataset.fields["text"].process([self.glove_dataset[idx].text])[0].T[0]
        # print(ids)1
        tweet = [self.glove_embedding_lookup.vocab.itos[i] if i != 1 else "" for i in ids]
        # print(tweet)

        # todo uncomment
        # elmo_embeddings = self.elmo_embedding_lookup.embed([tweet]).squeeze(0)

        # print(elmo_embeddings)
        # print(self.glove_dataset.fields["text"].process([self.glove_dataset[idx].text])[0])
        # print("elmo shape:{}".format(elmo_embeddings.shape))

        glove_embeddings = self.glove_embedding_lookup.vocab.vectors[self.glove_dataset.fields["text"].process([self.glove_dataset[idx].text])[0].T].squeeze(0)
        syngcn_embeddings = self.syngcn_embedding_lookup.vocab.vectors[self.syngcn_dataset.fields["text"].process([self.glove_dataset[idx].text])[0].T].squeeze(0)
        # print("glove shpae:{}".format(glove_embeddings.shape))
        # print("syn shape:{}".format(syngcn_embeddings.shape))

        # embeddings = torch.cat((glove_embeddings, syngcn_embeddings, torch.tensor(elmo_embeddings)), 1)
        embeddings = torch.cat((glove_embeddings, syngcn_embeddings), 1)
        # print(embeddings.shape)

        # print(embeddings)
        label = self.glove_dataset[idx].label

        # return {"embeddings": torch.tensor(embeddings), "labels": torch.tensor(labels)}
        return {"embeddings": embeddings.type(torch.FloatTensor),
                "label": torch.tensor(int(label))}, tweet


class BertTwitterDataset(Dataset):
    """Twitter dataset."""

    def __init__(self, csv_file=None, tweet_data_frame=None, transform=None, sentence_length_cut=40):
        """
        Args:
            csv_file (string): Path to the csv file with twitter files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if csv_file is not None:
          self.tweet_data_frame = pd.read_csv(csv_file)
        elif tweet_data_frame is not None:
          self.tweet_data_frame = tweet_data_frame
        else:
          #abcd
          pass

        self.transform = transform

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tweets = self.tweet_data_frame['text']
        self.labels = self.tweet_data_frame['label']
        self.tweet_list = self.sentences_from_df()
        self.tokenized_tweets = torch.LongTensor(self.tokenize_sentences(self.tweet_list, self.tokenizer, max_seq_len=sentence_length_cut))


    def __len__(self):
        return len(self.tweet_data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

   
        tweet = self.tokenized_tweets[idx]
        label = self.labels[idx]
        sample = {'text': tweet, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def sentences_from_df(self):
        sentences = []
        for i in range(len(self.tweets)):
          sentences.append(str(self.tweets.loc[i]))
        return sentences

    def tokenize_sentences(self,sentences, tokenizer, max_seq_len=40):
      """Encode sentences for using with BERT"""
      tokenized_sentences = []

      for sentence in sentences:
          tokenized_sentence = tokenizer.encode(
                              sentence,                  # Sentence to encode.
                              max_length = max_seq_len,  # Truncate all sentences.
                              pad_to_max_length=True,    # padding with zeros
                              truncation=True
                      )
          tokenized_sentences.append(tokenized_sentence)

      return tokenized_sentences
