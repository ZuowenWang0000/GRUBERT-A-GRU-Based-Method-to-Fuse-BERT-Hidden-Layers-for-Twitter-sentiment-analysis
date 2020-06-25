import torch
import torch.nn as nn
import torch.nn.functional as F

class LstmModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, word_rnn_size=None, word_rnn_layers=None, word_att_size=None, dropout=0.5, device=None):
        super().__init__()
        emb_sum_sizes = sum(emb_sizes_list)
        self.emb_weights = torch.nn.Parameter(torch.ones([emb_sum_sizes], requires_grad=True))
        self.lstm = nn.LSTM(emb_sum_sizes, word_rnn_size, bidirectional=True)
        self.lin = nn.Linear(2*word_rnn_size, n_classes)
    
    def forward(self, embeddings):
        embeddings = torch.mul(self.emb_weights, embeddings)  # element-wise multiplication
        x, _ = self.lstm(embeddings.permute(1, 0, 2))
        x = F.relu(x.permute(1, 0, 2))
        x = self.lin(x)
        x = x.sum(dim=1)
        return x, None, None