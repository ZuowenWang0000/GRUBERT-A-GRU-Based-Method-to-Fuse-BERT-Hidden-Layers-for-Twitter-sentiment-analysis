import torch
import numpy as np

def initialize_embeddings(embedding, device, fine_tune_embeddings=False):
    """
    Initializes embeddings for a given embedding string (see README for a list of supported embeddings)
    :param embedding: string indicating the embedding
    :param device: device on which to initialize embedding
    :param fine_tune_embeddings: whether fine-tuning should be enabled (note: this param should only be true
        if calling initialize_embeddings from inside a model)
    """
    if embedding in ["flair", "bert", "elmo", "elmo-only", "glove-only", "syngcn-only", "glove-syngcn", "twitter-only"]:
        import flair
        from flair.datasets import CSVClassificationDataset
        from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings, StackedEmbeddings

        if embedding.startswith("gs-"):
            glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
            syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
            embeddings_list = [glove_embedding, syngcn_embedding]

        if embedding == "gs-flair":
            print("[flair] initializing GS-Flair embeddings", flush=True)
            embeddings_list += [FlairEmbeddings("mix-forward", chars_per_chunk=128, fine_tune=fine_tune_embeddings), FlairEmbeddings("mix-backward", chars_per_chunk=128, fine_tune=fine_tune_embeddings)]
        elif embedding == "flair":
            print("[flair] initializing Flair embeddings", flush=True)
            embeddings_list = [FlairEmbeddings("mix-forward", chars_per_chunk=128, fine_tune=fine_tune_embeddings), FlairEmbeddings("mix-backward", chars_per_chunk=128, fine_tune=fine_tune_embeddings)]
        elif embedding == "gs-bert":
            print("[flair] initializing GS-Bert embeddings", flush=True)
            embeddings_list += [TransformerWordEmbeddings('bert-base-uncased', layers='-1', fine_tune=fine_tune_embeddings)]
        elif embedding == "gs-elmo":
            print("[flair] initializing ELMo embeddings", flush=True)
            embeddings_list += [ELMoEmbeddings(model="original", embedding_mode="top")]
        elif embedding == "elmo":
            print("[flair] initializing ELMo embeddings", flush=True)
            embeddings_list = [ELMoEmbeddings(model="original", embedding_mode="top")]
        elif embedding == "glove":
            print("[flair] initializing Glove embeddings", flush=True)
            embeddings_list = [WordEmbeddings("../embeddings/glove.6B.300d.gensim")]
        elif embedding == "syngcn":
            print("[flair] initializing SynGCN only embeddings", flush=True)
            embeddings_list = [WordEmbeddings("../embeddings/syngcn.gensim")]
        elif embedding == "twitter":
            print("[flair] initializing twitter only embeddings", flush=True)
            embeddings_list = [WordEmbeddings("en-twitter")]
        elif embedding == "gs-only":
            print("[flair] initializing Glove + SynGCN only embeddings", flush=True)
            embeddings_list = [glove_embedding, syngcn_embedding]

        return StackedEmbeddings(embeddings=embeddings_list).to(device)
    
    elif embedding in ["bert-mix", "bert-base", "bert-last-four"]:
        from transformers import BertModel
        embedder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in embedder.parameters():
            param.requires_grad = fine_tune_embeddings
        return embedder
    elif embedding in ["roberta-mix"]:
        from transformers import RobertaModel
        embedder = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
        for param in embedder.parameters():
            param.requires_grad = fine_tune_embeddings
        return embedder
    
    else:
        raise NotImplementedError("Unsupported embedding: " + embedding)


def prepare_embeddings_flair(sentences, embedder, device, params):
    """
    Performs embedding for embeddings provided by the Flair NLP library.
    :param sentences: sentences returned by Flair data loader
    :param embedder: embedder to perform embedding with
    :param device: device on which to perform embedding
    :param params: config
    """
    embedder.embed(sentences)  # Perform embedding

    # Perform zero-padding up to the max sentence length in the batch
    lengths = [len(sentence.tokens) for sentence in sentences]
    longest_token_sequence_in_batch: int = max(lengths)
    pre_allocated_zero_tensor = torch.zeros(
        embedder.embedding_length * longest_token_sequence_in_batch,
        dtype=torch.float,
        device=device,
    )
    all_embs = list()
    for sentence in sentences:
        all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
        nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
        if nb_padding_tokens > 0:
            t = pre_allocated_zero_tensor[:embedder.embedding_length * nb_padding_tokens]
            all_embs.append(t)
    embeddings = torch.cat(all_embs).view([len(sentences), longest_token_sequence_in_batch, embedder.embedding_length])

    # Get labels
    labels = torch.as_tensor(np.array([int(s.labels[0].value) for s in sentences]))

    return embeddings.to(device), labels.to(device)


def prepare_embeddings_bert_mix(data, embedder, device, params):
    """
    Performs embedding for embeddings provided by the Flair NLP library.
    :param data: data returned by data loader
    :param embedder: embedder to perform embedding with
    :param device: device on which to perform embedding
    :param params: config
    """
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))  # Perform embedding
    labels = labels.to(device)

    # Calculate how many BERT hidden layers each GRU is combining
    num_combined_per_gru = int(12 / params.model.num_grus)

    # Concatenate the BERT hidden layers accordingly
    h = [torch.cat(embeddings[2][i*num_combined_per_gru+1 : (i+1)*num_combined_per_gru+1], 2) for i in range(params.model.num_grus)]
    return h, labels


def prepare_embeddings_roberta_mix(data, embedder, device, params):
    # Exactly the same as for BERT, provided for convenience
    return prepare_embeddings_bert_mix(data, embedder, device, params)


def prepare_embeddings_bert_base(data, embedder, device, params):
    """
    Performs embedding for embeddings provided by the Flair NLP library.
    :param data: data returned by data loader
    :param embedder: embedder to perform embedding with
    :param device: device on which to perform embedding
    :param params: config
    """
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))  # Perform embedding
    labels = labels.to(device)
    h2 = embeddings[2][12]  # Get the last layer of the BERT model
    return [h2], labels


def prepare_embeddings_bert_last_four(data, embedder, device, params):
    """
    Performs embedding for embeddings provided by the Flair NLP library.
    :param data: data returned by data loader
    :param embedder: embedder to perform embedding with
    :param device: device on which to perform embedding
    :param params: config
    """
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))  # Perform embedding
    labels = labels.to(device)
    h2 = torch.cat(embeddings[2][9:13], 2)  # Concatenate the last 4 layers of the BERT model
    return [h2], labels
