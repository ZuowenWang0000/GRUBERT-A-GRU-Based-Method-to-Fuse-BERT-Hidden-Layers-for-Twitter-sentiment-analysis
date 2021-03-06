import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from embeddings import initialize_embeddings


class BertMixModel(nn.Module):
    """
    Model which uses two layers of GRUs to combine the BERT hidden layers. The first layer combines 
    groups of layers, and the second layer combines the outputs of the first layer.
    """
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        # Initialize embedder for use by training loop
        self.embedder = initialize_embeddings(model_config.embedding_type, model_config.device, fine_tune_embeddings=model_config.fine_tune_embeddings)
        self.embedder = self.embedder.to(self.device)

        self.num_grus = model_config.num_grus
        assert 12 % self.num_grus == 0
        self.num_combined_per_gru = int(12 / self.num_grus)  # Number of BERT hidden layers combined per GRU

        # Initialize combining GRUs (first layer)
        self.grus = [nn.GRU(self.num_combined_per_gru * 768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True) for _ in range(self.num_grus)]

        # Initialize combining GRU (second layer)
        self.gru = nn.GRU(2 * self.num_grus * model_config.gru_hidden_size, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers, bidirectional=True)
        
        # Initialize classifier (after document embedding is complete using two layers of GRUs)
        self.classifier = nn.Sequential(
            nn.Linear(2*model_config.gru_hidden_size, model_config.linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=model_config.dropout),
            nn.Linear(model_config.linear_hidden_size, n_classes),
        )

        # Initialize layers
        for layer in self.classifier:
            if (isinstance(layer,nn.Linear)):
                torch.nn.init.xavier_normal_(layer.weight)
    
    def forward(self, embeddings):
        # Assumes that embeddings have already been computed by the training loop
        temp = [embeddings[i].to(self.device).permute(1, 0, 2) for i in range(self.num_grus)]  # Bring into correct order
        out = [self.grus[i].to(self.device)(temp[i])[0] for i in range(self.num_grus)]  # First layer of GRUs

        # Second layer
        x1 = torch.cat(out, 2).to(self.device)
        x, _ = self.gru(x1)

        # Classifier
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return {"logits": x}


class BertBaseModel(nn.Module):
    """
    Model using only the last hidden layer of BERT.
    """
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        self.embedder = initialize_embeddings(model_config.embedding_type, model_config.device, fine_tune_embeddings=model_config.fine_tune_embeddings)
        self.embedder = self.embedder.to(self.device)

        self.gru1 = nn.GRU(768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
                           bidirectional=True)
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
        layer = embeddings[0]
        x = layer.to(self.device).permute(1, 0, 2)
        o1, _ = self.gru1(x)

        x1 = o1.to(self.device)

        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)
        return {"logits": x}


class BertLastFourModel(nn.Module):
    """
    Model using only the concatenation of the last four BERT layers.
    """
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        self.embedder = initialize_embeddings(model_config.embedding_type, model_config.device, fine_tune_embeddings=model_config.fine_tune_embeddings)
        self.embedder = self.embedder.to(self.device)

        self.gru1 = nn.GRU(4*768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
                           bidirectional=True)
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
        # embeddings = [layer]
        layer = embeddings[0]
        x = layer.to(self.device).permute(1, 0, 2)
        o1, _ = self.gru1(x)

        x1 = o1.to(self.device)

        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)
        return {"logits": x}


class BertWSModel(nn.Module):
    """
    The same as BertMixModel, but using a weight-sharing GRU in the first layer insead of separate GRUs.
    """
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        self.embedder = initialize_embeddings(model_config.embedding_type, model_config.device, fine_tune_embeddings=model_config.fine_tune_embeddings)
        self.embedder = self.embedder.to(self.device)

        self.num_grus = model_config.num_grus
        assert 12 % self.num_grus == 0
        self.num_combined_per_gru = int(12 / self.num_grus)

        self.gru1 = nn.GRU(self.num_combined_per_gru * 768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
                           bidirectional=True)
        self.gru = nn.GRU(2 * self.num_grus * model_config.gru_hidden_size, model_config.gru_hidden_size,
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
        temp = [embeddings[i].to(self.device).permute(1, 0, 2) for i in range(self.num_grus)]
        out = [self.gru1.to(self.device)(temp[i])[0] for i in range(self.num_grus)]

        x1 = torch.cat(out, 2).to(self.device)
        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return {"logits": x}


class BertMixLinearModel(nn.Module):
    """
    The same as BertMixModel, but using linear layers instead of GRUs in the first layer.
    """
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        self.embedder = initialize_embeddings(model_config.embedding_type, model_config.device, fine_tune_embeddings=model_config.fine_tune_embeddings)
        self.embedder = self.embedder.to(self.device)

        self.linear = nn.Linear(4 * 768, 2 * model_config.gru_hidden_size)
        torch.nn.init.xavier_normal_(self.linear.weight)

        self.gru = nn.GRU(6 * model_config.gru_hidden_size, model_config.gru_hidden_size,
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
        # embeddings = [layer14, layer58, layer912]
        layer14 = embeddings[0]
        layer58 = embeddings[1]
        layer912 = embeddings[2]
        x14 = layer14.to(self.device).permute(1, 0, 2)
        x58 = layer58.to(self.device).permute(1, 0, 2)
        x912 = layer912.to(self.device).permute(1, 0, 2)

        o1 = self.linear(x14)
        o2 = self.linear(x58)
        o3 = self.linear(x912)

        x1 = torch.cat([o1, o2, o3], 2).to(self.device)

        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return {"logits": x}

class BertMixLSTMModel(nn.Module):
    """
    The same as BertMixModel, but using LSTMs for the first layer instead of GRUs.
    """
    def __init__(self, n_classes, model_config):
        super().__init__()
        self.device = model_config.device

        self.embedder = initialize_embeddings(model_config.embedding_type, model_config.device, fine_tune_embeddings=model_config.fine_tune_embeddings)
        self.embedder = self.embedder.to(self.device)

        self.lstm = nn.LSTM(4 * 768, model_config.gru_hidden_size, num_layers=model_config.num_gru_layers,
                            bidirectional=True)
        self.gru = nn.GRU(6 * model_config.gru_hidden_size, model_config.gru_hidden_size,
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
        # embeddings = [layer14, layer58, layer912]
        layer14 = embeddings[0]
        layer58 = embeddings[1]
        layer912 = embeddings[2]
        x14 = layer14.to(self.device).permute(1, 0, 2)
        x58 = layer58.to(self.device).permute(1, 0, 2)
        x912 = layer912.to(self.device).permute(1, 0, 2)

        o1, _ = self.lstm(x14)
        o2, _ = self.lstm(x58)
        o3, _ = self.lstm(x912)

        x1 = torch.cat([o1, o2, o3], 2).to(self.device)

        x, _ = self.gru(x1)
        x = F.relu(x.permute(1, 0, 2))
        x = self.classifier(x)
        x = x.sum(dim=1)

        return {"logits": x}

