import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class FlairModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, model_config):
        super().__init__()
        emb_sum_sizes = sum(emb_sizes_list)
        # self.emb_weights = torch.nn.Parameter(torch.ones([emb_sum_sizes], requires_grad=True))
        # self.lin = nn.Linear(2 * model_config.lstm_hidden_size, n_classes)
        #todo how do we combine embeddings? linear layer or GRU?

        # self.dropout = nn.Dropout(model_config.dropout)

        self.device = eval(model_config.device)

        # self.gru1 = nn.GRU(4*768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True)
        # self.gru2 = nn.GRU(4*768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True)
        # self.gru3 = nn.GRU(4*768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True)

        self.gru = nn.GRU(emb_sum_sizes, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(2*model_config.gru_hidden_size, model_config.linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=model_config.dropout),
            nn.Linear(model_config.linear_hidden_size, n_classes),
        )
        for layer in self.classifier:
            if (isinstance(layer,nn.Linear)):
                torch.nn.init.xavier_normal_(layer.weight)

        for param in self.embedder.parameters():
            param.requires_grad = True # todo: replace this by fine_tune？ ZUOWEN
    
    def forward(self, embeddings):
        # embeddings = [layer14, layer58, layer912]
        x, _ = self.lstm(embeddings.permute(1, 0, 2))

        layer14 = embeddings[0]
        layer58 = embeddings[1]
        layer912 = embeddings[2]
        x14 = layer14.to(self.device).permute(1,0,2)
        x58 = layer58.to(self.device).permute(1,0,2)
        x912 = layer912.to(self.device).permute(1,0,2)

        # todo check this gru1
        o1,_ = self.gru1(x14)
        o2,_ = self.gru1(x58)
        o3,_ = self.gru1(x912)

        x1 = torch.cat([o1,o2,o3],2).to(self.device)

        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return {"logits": x}