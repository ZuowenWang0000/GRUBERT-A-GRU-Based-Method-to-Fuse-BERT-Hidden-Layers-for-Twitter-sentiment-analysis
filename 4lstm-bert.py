import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.data import Iterator



import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

from collections import namedtuple
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer


import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from transformers import BertModel


test_path="dataset/" # location where test data resides
preprocessed_path = "dataset/" # location where data_preprocessed folder resides
save_path = "dataset/" # folder where you want to save files created by this notebook


Item = namedtuple('Item','text label')
items = []

with open(preprocessed_path + "train_pos.txt") as f:
    for line in f:
        l = line.rstrip('\n')
        items.append(Item(l, 1))

with open(preprocessed_path + "train_neg.txt") as f:
    for line in f:
        l = line.rstrip('\n')
        items.append(Item(l, 0))

df = pd.DataFrame.from_records(items, columns=['text', 'label'])



df.to_csv(save_path + "train_all.csv",index=False,header=True)

main_df = pd.read_csv(save_path + "train_all.csv")
train , val = train_test_split(main_df,stratify=main_df['label'],test_size=0.2,shuffle=True)
train.to_csv(save_path + "train_split.csv",header=True,index=False)
val.to_csv(save_path + "val_split.csv",header=True,index=False)


class TwitterDataset(Dataset):
    """Twitter dataset."""

    def __init__(self, csv_file=None, tweet_data_frame=None, transform=None):
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
            # abcd
            pass

        self.transform = transform

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tweets = self.tweet_data_frame['text']
        self.labels = self.tweet_data_frame['label']
        self.tweet_list = self.sentences_from_df()
        self.tokenized_tweets = torch.LongTensor(self.tokenize_sentences(self.tweet_list, self.tokenizer))

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

    def tokenize_sentences(self, sentences, tokenizer, max_seq_len=40):
        """Encode sentences for using with BERT"""
        tokenized_sentences = []

        for sentence in sentences:
            tokenized_sentence = tokenizer.encode(
                sentence,  # Sentence to encode.
                max_length=max_seq_len,  # Truncate all sentences.
                pad_to_max_length=True  # padding with zeros
            )
            tokenized_sentences.append(tokenized_sentence)

        return tokenized_sentences



train_data = TwitterDataset(csv_file=save_path + "train_split.csv")

val_data = TwitterDataset(csv_file=save_path + "val_split.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_workers = 4


class MyBertModel(LightningModule):
    def __init__(self,device):
        super().__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = self.model.to(device)
        self.model.eval()
        #self.lstm = nn.LSTM(768*12, 768,bidirectional=True)
        #self.lin = nn.LSTM(768*2,768)
        self.lstm1 = nn.LSTM(768*4,100,bidirectional=True)
        self.lstm2 = nn.LSTM(768*4,100,bidirectional=True)
        self.lstm3 = nn.LSTM(768*4,100,bidirectional=True)
        self.lstm = nn.LSTM(100*2*3,100,bidirectional=True)
        self.lin1 = nn.Linear(2*100, 100)
        self.lin2 = nn.Linear(100,2)

        self.loss_function = nn.CrossEntropyLoss()
        #self.train_correction_matrix  = {}
        #self.val_correction_matrix = {}


    def forward(self, x1, x2, x3):


        #x, _ = self.lstm(x1)
        #x = F.relu(x.permute(1,0,2))
        #x = self.lin(x)
        x1 = x1.to(device).permute(1,0,2)
        x2 = x2.to(device).permute(1,0,2)
        x3 = x3.to(device).permute(1,0,2)

        x1, _ = self.lstm1(x1)
        x1 = F.relu(x1.permute(1, 0, 2))

        x2, _ = self.lstm2(x2)
        x2 = F.relu(x1.permute(1, 0, 2))

        x3, _ = self.lstm3(x3)
        x3 = F.relu(x3.permute(1, 0, 2))
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        x2 = x2.permute(1,0,2)
        x = torch.cat([x1,x2,x3],2).permute(1,0,2).to(device)
        #print(x.shape)
        x, _ = self.lstm(x)
        x = F.relu(x.permute(1, 0, 2))

        x = self.lin1(x)
        x = F.relu(x)
        x = nn.Dropout(0.5)(x)
        x = self.lin2(x)
        x = x.sum(dim=1)


        return x

    def prepare_data(self):
        pass

    def svd_embed(self, batch, batch_idx):
        x = batch["text"]

        x = x.to(device)

        with torch.no_grad():
          embeddings = self.model(input_ids=x)

          group = embeddings[2]

          em_len = group[1].shape[2]
          twelve_hidden = torch.zeros((group[1].shape[0],group[1].shape[1],12*em_len))
          rez = torch.zeros((group[1].shape[0],12,12*em_len))


          for i in range(group[1].shape[0]):
              twelve_hidden[i,:,:em_len] = group[1][i,:,:]
              twelve_hidden[i,:,em_len:em_len*2] = group[2][i,:,:]
              twelve_hidden[i,:,em_len*2:em_len*3] = group[3][i,:,:]
              twelve_hidden[i,:,em_len*3:em_len*4] = group[4][i,:,:]

              twelve_hidden[i,:,em_len*4:em_len*5] = group[5][i,:,:]
              twelve_hidden[i,:,em_len*5:em_len*6] = group[6][i,:,:]
              twelve_hidden[i,:,em_len*6:em_len*7] = group[7][i,:,:]
              twelve_hidden[i,:,em_len*7:em_len*8] = group[8][i,:,:]

              twelve_hidden[i,:,em_len*8:em_len*9] = group[9][i,:,:]
              twelve_hidden[i,:,em_len*9:em_len*10] = group[10][i,:,:]
              twelve_hidden[i,:,em_len*10:em_len*11] = group[11][i,:,:]
              twelve_hidden[i,:,em_len*11:] = group[12][i,:,:]

          for i in range(group[1].shape[0]):
            t = twelve_hidden[i]
            rez[i,:,:] = torch.svd(t)


        return rez


    def make_groups(self,batch,batch_idx):
        x = batch["text"]

        x = x.to(device)

        with torch.no_grad():
          embeddings = self.model(input_ids=x)

        hidden = embeddings[2]

        em_len = hidden[1].shape[2]
        group1 = torch.zeros((hidden[1].shape[0],hidden[1].shape[1],4*em_len))
        group2 = torch.zeros((hidden[1].shape[0],hidden[1].shape[1],4*em_len))
        group3 = torch.zeros((hidden[1].shape[0],hidden[1].shape[1],4*em_len))


        for i in range(hidden[1].shape[0]):
          group1[i,:,:em_len] = hidden[1][i,:,:]
          group1[i,:,em_len:2*em_len] = hidden[2][i,:,:]
          group1[i,:,2*em_len:3*em_len] = hidden[3][i,:,:]
          group1[i,:,3*em_len:] = hidden[4][i,:,:]

          group2[i,:,:em_len] = hidden[5][i,:,:]
          group2[i,:,em_len:2*em_len] = hidden[6][i,:,:]
          group2[i,:,2*em_len:3*em_len] = hidden[7][i,:,:]
          group2[i,:,3*em_len:4*em_len] = hidden[8][i,:,:]

          group3[i,:,:em_len] = hidden[9][i,:,:]
          group3[i,:,em_len:2*em_len] = hidden[10][i,:,:]
          group3[i,:,2*em_len:3*em_len] = hidden[11][i,:,:]
          group3[i,:,3*em_len:4*em_len] = hidden[12][i,:,:]

        return group1,group2,group3







    def train_dataloader(self):
        #self.train_correction_matrix = train_data.to_be_averaged
        return DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            #collate_fn=custom_collate,
            shuffle=False
            )

    def val_dataloader(self):
        #self.val_correction_matrix = val_data.to_be_averaged
        return DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            #collate_fn=custom_collate,
            shuffle=False
            )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        """
        x,y = batch["text"],batch["label"]

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
          embeddings = self.model(input_ids=x)
          hidden_states = embeddings[2]
          len(hidden_states)
          em_len = hidden_states[0].shape[2]
          four_hidden = torch.zeros((hidden_states[0].shape[0],hidden_states[0].shape[1],4*hidden_states[0].shape[2]))
          for i in range(hidden_states[0].shape[0]):
              four_hidden[i,:,:em_len] = hidden_states[0][i,:,:]
              four_hidden[i,:,em_len:em_len*2] = hidden_states[1][i,:,:]
              four_hidden[i,:,em_len*2:em_len*3] = hidden_states[2][i,:,:]
              four_hidden[i,:,em_len*3:em_len*4] = hidden_states[3][i,:,:]

        """
        #x = batch["text"]

        #x = x.to(device)

        #with torch.no_grad():
        #  embeddings = self.model(input_ids=x)
        #hidden = torch.stack(embeddings[2][1:],0).squeeze(1).mean(2).permute(1,0,2).to(device)
        group1,group2,group3 = self.make_groups(batch,batch_idx)

        #x = self.make_pair_embed(batch,batch_idx)
        y = batch["label"]
        y = y.to(device)
        # feed embeddings in network
        y_hat = self.forward(group1,group2,group3)
        loss = self.loss_function(y_hat, y)

        # calculate accuracy for batch
        accuracy = (y_hat.argmax(-1) == y).float().mean()

        tensorboard_logs = {'train_loss': loss,'train_accuracy':accuracy}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        x,y = batch["text"],batch["label"]

        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
          embeddings = self.model(input_ids=x)
          hidden_states = embeddings[2]
          em_len = hidden_states[0].shape[2]
          four_hidden = torch.zeros((hidden_states[0].shape[0],hidden_states[0].shape[1],4*hidden_states[0].shape[2]))
          for i in range(hidden_states[0].shape[0]):
              four_hidden[i,:,:em_len] = hidden_states[0][i,:,:]
              four_hidden[i,:,em_len:em_len*2] = hidden_states[1][i,:,:]
              four_hidden[i,:,em_len*2:em_len*3] = hidden_states[2][i,:,:]
              four_hidden[i,:,em_len*3:em_len*4] = hidden_states[3][i,:,:]

        """

        #x = batch["text"]

        #x = x.to(device)

        #with torch.no_grad():
        #  embeddings = self.model(input_ids=x)
        #hidden = torch.stack(embeddings[2][1:],0).squeeze(1).mean(2).permute(1,0,2).to(device)
                # feed embeddings in network
        group1,group2,group3 = self.make_groups(batch,batch_idx)
        y = batch["label"]
        y = y.to(device)
        y_hat = self.forward(group1,group2,group3)
        loss = self.loss_function(y_hat, y)

        # calculate accuracy for batch
        accuracy = (y_hat.argmax(-1) == y).float().mean()

        tensorboard_logs = {'val_loss': loss,'val_acc':accuracy}
        return {'val_loss': loss, 'val_acc':accuracy, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        return {
          'val_loss': avg_loss,
          'val_acc': avg_acc,
          'progress_bar':{'val_loss': avg_loss, 'val_acc': avg_acc }}


debug = False
model = MyBertModel(device).cuda()

class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('Trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')


if(debug):
  trainer = Trainer(
      fast_dev_run=True,
      max_epochs=1,
      )
else:
  logger = TensorBoardLogger('tb_logs', name='my_model')
  trainer = Trainer(
      logger=logger,
      gpus=1, # run on one GPU
      max_epochs=20,
      callbacks = [MyPrintingCallback()],
      weights_summary='full'
  )


trainer.fit(model)





df = pd.read_csv(test_path + "test_cleaned.csv")



test_data = TwitterDataset(tweet_data_frame=df)

def classify_tweet(tweet):
    x = tweet["text"]
    print(x.shape)
    x = x.unsqueeze(0)

    x = x.to(device)
    with torch.no_grad():
        print("HERE")
        print(x.shape)
        embeddings = model.model(input_ids=x)
        print("HER2E")

    hidden = embeddings[2]
    em_len = hidden[1].shape[2]
    group1 = torch.zeros((hidden[1].shape[0], hidden[1].shape[1], 4 * em_len))
    group2 = torch.zeros((hidden[1].shape[0], hidden[1].shape[1], 4 * em_len))
    group3 = torch.zeros((hidden[1].shape[0], hidden[1].shape[1], 4 * em_len))

    for i in range(hidden[1].shape[0]):
        group1[i, :, :em_len] = hidden[1][i, :, :]
        group1[i, :, em_len:2 * em_len] = hidden[2][i, :, :]
        group1[i, :, 2 * em_len:3 * em_len] = hidden[3][i, :, :]
        group1[i, :, 3 * em_len:] = hidden[4][i, :, :]

        group2[i, :, :em_len] = hidden[5][i, :, :]
        group2[i, :, em_len:2 * em_len] = hidden[6][i, :, :]
        group2[i, :, 2 * em_len:3 * em_len] = hidden[7][i, :, :]
        group2[i, :, 3 * em_len:4 * em_len] = hidden[8][i, :, :]

        group3[i, :, :em_len] = hidden[9][i, :, :]
        group3[i, :, em_len:2 * em_len] = hidden[10][i, :, :]
        group3[i, :, 2 * em_len:3 * em_len] = hidden[11][i, :, :]
        group3[i, :, 3 * em_len:4 * em_len] = hidden[12][i, :, :]

    y_hat = model(group1, group2, group3).argmax().item()
    return 2 * (y_hat - 0.5)  # convert back to -1,1 labels




Item = namedtuple('Item','Id Prediction')
items = []

for i in range(len(test_data)):
    tweet = test_data[i]
    pred = classify_tweet(tweet)
    print(i,pred)
    items.append(Item(i+1,int(pred)))

predictions = pd.DataFrame.from_records(items, columns=['Id','Prediction'])

print(predictions)

predictions.Prediction.value_counts()
predictions.to_csv(save_path + "submission_bert_4lstm.csv",index = False)


# Start tensorboard.
#%reload_ext tensorboard
#%tensorboard --logdir tb_logs/
