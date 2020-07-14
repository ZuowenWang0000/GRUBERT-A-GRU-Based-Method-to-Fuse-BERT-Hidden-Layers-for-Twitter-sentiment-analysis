import torch
import torch.nn as nn
import torch.nn.functional as F

class LstmModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, model_config):
        super().__init__()
        emb_sum_sizes = sum(emb_sizes_list)
        self.emb_weights = torch.nn.Parameter(torch.ones([emb_sum_sizes], requires_grad=True))
        self.lstm = nn.LSTM(emb_sum_sizes, model_config.lstm_hidden_size, num_layers=model_config.num_lstm_layers, bidirectional=True, dropout=model_config.dropout)
        self.lin = nn.Linear(2*model_config.lstm_hidden_size, n_classes)
        self.dropout = nn.Dropout(model_config.dropout)
    
    def forward(self, embeddings):
        embeddings = torch.mul(self.emb_weights, embeddings)  # element-wise multiplication
        # embeddings = self.dropout(embeddings)
        x, _ = self.lstm(embeddings.permute(1, 0, 2))
        x = F.relu(x.permute(1, 0, 2))
        x = self.dropout(x)
        x = self.lin(x)
        x = x.sum(dim=1)
        return {"logits": x}