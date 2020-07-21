import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class GSFlairMixModel(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, model_config):
        super().__init__()
        self.device = eval(model_config.device)

        # self.embedder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.embedder = self.embedder.to(self.device)

        emb_sum_sizes = sum(emb_sizes_list)
        self.gru1 = nn.GRU(emb_sum_sizes, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
                           bidirectional=True)
        # self.gru2 = nn.GRU(4 * 768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
        #                    bidirectional=True)
        # self.gru3 = nn.GRU(4 * 768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
        #                    bidirectional=True)
        # todo warning: this gru's input is smaller (3 times) than the bert-mix Zuowen
        self.gru = nn.GRU(2 * model_config.gru_hidden_size, model_config.gru_hidden_size,
                          num_layers=model_config.num_gru_layers, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * model_config.gru_hidden_size, model_config.linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=model_config.dropout),
            nn.Linear(model_config.linear_hidden_size, n_classes),
        )
        for layer in self.classifier:
            if (isinstance(layer, nn.Linear)):
                torch.nn.init.xavier_normal_(layer.weight)


    def forward(self, embeddings):
        x = embeddings.to(self.device).permute(1, 0, 2)

        o1, _ = self.gru1(x)

        x, _ = self.gru(o1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return {"logits": x}
