import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

class AttentionNetwork(LightningModule):
    def __init__(self, train_dataloader, test_dataloader, n_classes, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, classifier_size, dropout=0.5):
        super(AttentionNetwork, self).__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout)
        self.fc = nn.Linear(2 * word_rnn_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, glove_seq, syngcn_seq, elmo_seq):
        # tweet_embedding = self.word_attention()  # TODO
        # TEMP TEMP
        sentence_embedding, word_alphas = self.word_attention(glove_seq)
        score = self.fc(self.dropout(sentence_embedding))
        return score, word_alphas

    def train_dataloader(self):
        return self.train_dataloader

    def test_dataloader(self):
        return self.test_dataloader

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch.text[0].T, batch.label
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        # calculate accuracy for batch
        _, predicted = torch.max(y_hat,1)
        accuracy = (predicted == y).sum().type(torch.DoubleTensor) / len(y)


        tensorboard_logs = {'train_loss': loss,'train_accuracy':accuracy}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch.text[0].T, batch.label
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        
        # calculate accuracy for batch
        _, predicted = torch.max(y_hat,1)
        accuracy = (predicted == y).sum().type(torch.DoubleTensor) / len(y)

        tensorboard_logs = {'val_loss': loss, 'val_acc':accuracy}
        return {'val_loss': loss,'val_acc':accuracy, 'log':tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        return {
          'val_loss': avg_loss,
          'val_acc': avg_acc, 
          'progress_bar':{'val_loss': avg_loss, 'val_acc': avg_acc }}


class WordAttention(LightningModule):
    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        super(WordAttention, self).__init__()
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, combined_seq):
        combined_seq, _ = self.word_rnn(combined_seq)
        att_w = self.word_attention(combined_seq.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Calculate softmax values
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Find sentence embedding
        sentence = combined_seq * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentence = sentence.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentence, word_alphas