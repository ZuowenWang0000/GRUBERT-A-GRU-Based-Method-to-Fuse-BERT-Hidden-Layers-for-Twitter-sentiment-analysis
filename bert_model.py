import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertSentimentModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, word_rnn_size=None, word_rnn_layers=None, word_att_size=None, dropout=0.5, device=None):
        super().__init__()
        # emb_sum_sizes = sum(emb_sizes_list)
        # self.emb_weights = torch.nn.Parameter(torch.ones([emb_sum_sizes], requires_grad=True))
        # self.lstm = nn.LSTM(emb_sum_sizes, word_rnn_size, bidirectional=True)
        # self.lin = nn.Linear(2*word_rnn_size, n_classes)
        self.device = device
        # self.save_hyperparameters()
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = self.model.to(device)

        self.gru1 = nn.GRU(4*768,100,bidirectional=True)
        self.gru2 = nn.GRU(4*768,100,bidirectional=True)
        self.gru3 = nn.GRU(4*768,100,bidirectional=True)

        self.gru = nn.GRU(6*100, 100,bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(2*100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 2),
        )
        for layer in self.classifier:
          if(isinstance(layer,nn.Linear)):
            torch.nn.init.xavier_normal_(layer.weight)

        for param in self.model.parameters():
          param.requires_grad = True
    
    
    def forward(self, embeddings):
        # embeddings = [layer14, layer58, layer912]
        layer14 = embeddings[0]
        layer58 = embeddings[1]
        layer912 = embeddings[2]
        x14 = layer14.to(self.device).permute(1,0,2)
        x58 = layer58.to(self.device).permute(1,0,2)
        x912 = layer912.to(self.device).permute(1,0,2)

        o1,_ = self.gru1(x14)
        o2,_ = self.gru1(x58)
        o3,_ = self.gru1(x912)

        x1 = torch.cat([o1,o2,o3],2).to(self.device)

        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return x, _, _

        # embeddings = torch.mul(self.emb_weights, embeddings)  # element-wise multiplication
        # x, _ = self.lstm(embeddings.permute(1, 0, 2))
        # x = F.relu(x.permute(1, 0, 2))
        # x = self.lin(x)
        # x = x.sum(dim=1)
        # return x, None, self.emb_weights