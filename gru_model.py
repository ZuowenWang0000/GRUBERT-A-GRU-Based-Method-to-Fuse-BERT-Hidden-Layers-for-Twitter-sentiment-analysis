import torch
import torch.nn as nn
import torch.nn.functional as F

class GruModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, model_config):
        super().__init__()
        emb_sum_sizes = sum(emb_sizes_list)
        self.emb_weights = torch.nn.Parameter(torch.ones([emb_sum_sizes], requires_grad=True))
        self.gru = nn.GRU(emb_sum_sizes, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(2*model_config.gru_hidden_size, n_classes)
        self.dropout = nn.Dropout(model_config.dropout)
    
    def forward(self, embeddings):
        embeddings = torch.mul(self.emb_weights, embeddings)  # element-wise multiplication
        embeddings = self.dropout(embeddings)
        x, _ = self.gru(embeddings)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.lin(x)
        x = x.sum(dim=1)
        return x, None, self.emb_weights