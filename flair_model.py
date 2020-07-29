import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from embeddings import initialize_embeddings


class GSFlairMixModel(nn.Module):
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        self.embedder = initialize_embeddings(model_config.embedding_type, self.device, fine_tune_embeddings=model_config.fine_tune_embeddings)

        emb_sum_sizes = sum([e.embedding_length for e in self.embedder.embeddings])
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
