import torch
import torch.nn as nn
import torch.nn.functional as F

class LstmModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, word_rnn_size=None, word_rnn_layers=None, word_att_size=None, dropout=0.5, device=None):
        super().__init__()
        self.lstm = nn.LSTM(emb_sizes_list[0], word_rnn_size, bidirectional=True)
        self.lin = nn.Linear(2*word_rnn_size, n_classes)
    
    def forward(self, embeddings):
        x, _ = self.lstm(embeddings)
        x = F.relu(x)
        x = self.lin(x)
        x = x.sum(dim=1)
        return x, None, None