import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

class AttentionNetwork(nn.Module):
    def __init__(self, n_classes, emb_sizes, word_rnn_size, word_rnn_layers, word_att_size, dropout=0.5, batch_size=64):
        super(AttentionNetwork, self).__init__()
        with torch.no_grad():
            sum_sizes = torch.sum(emb_sizes)
        self.word_attention = WordAttention(sum_sizes, word_rnn_size, word_rnn_layers, word_att_size, dropout)
        self.fc = nn.Linear(2 * word_rnn_size, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.emb_weights = torch.ones([sum_sizes, 1])

    def forward(self, embeddings):
        # TODO embedding-wise weight control with mask
        embeddings = self.emb_weights * embeddings  # element-wise multiplication
        sentence_embedding, word_alphas = self.word_attention(embeddings[0])
        score = self.fc(self.dropout(sentence_embedding))
        return score, word_alphas, self.emb_weights


class WordAttention(LightningModule):
    def __init__(self, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
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