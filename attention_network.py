import torch
import torch.nn as nn

class AttentionNetwork(nn.Module):
    def __init__(self, n_classes, emb_sizes_list, model_config):
        super(AttentionNetwork, self).__init__()
        sum_sizes = sum(emb_sizes_list)
        self.word_attention = WordAttention(sum_sizes, model_config.word_rnn_size, model_config.word_rnn_layers, model_config.word_att_size, model_config.dropout)

        self.fc = nn.Linear(2 * model_config.word_rnn_size, n_classes)
        self.dropout = nn.Dropout(model_config.dropout)
        self.emb_weights = torch.nn.Parameter(torch.ones([sum_sizes], requires_grad=True))

        self.sum_sizes = sum_sizes

    def forward(self, embeddings):
        embeddings = torch.mul(self.emb_weights, embeddings)  # element-wise multiplication
        sentence_embedding, word_alphas = self.word_attention(embeddings)
        score = self.fc(self.dropout(sentence_embedding))
        return score, word_alphas, self.emb_weights


class WordAttention(nn.Module):
    def __init__(self, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        super(WordAttention, self).__init__()
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, combined_seq):
        # print("combined seq :{}".format(combined_seq.shape))
        combined_seq, _ = self.word_rnn(combined_seq.float().permute(1, 0, 2))
        # print("combined seq :{}".format(combined_seq.shape))
        combined_seq = combined_seq.permute(1, 0, 2)
        att_w = self.word_attention(combined_seq.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        # max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        # att_w = torch.exp(att_w - max_value)  # (n_words)
        word_alphas = self.softmax(att_w)

        # Calculate softmax values
        # word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))
        # print("word_alphas shape :{}".format(word_alphas.shape))
        # Find sentence embedding
        sentence = combined_seq * word_alphas  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        # print("sentence dim :{}".format(sentence.shape))
        sentence = sentence.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)
        # print("sentence dim :{}".format(sentence.shape))
        return sentence, word_alphas