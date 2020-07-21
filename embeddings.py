import torch
import numpy as np

def initialize_embeddings(embedding, device, fine_tune_embeddings=False):
    if embedding in ["flair", "bert", "elmo"]:
        import flair
        from flair.datasets import CSVClassificationDataset
        from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        embeddings_list = [glove_embedding, syngcn_embedding]

        if embedding == "flair":
            print("[flair] initializing Flair embeddings", flush=True)
            embeddings_list += [FlairEmbeddings("mix-forward", chars_per_chunk=64, fine_tune=fine_tune_embeddings), FlairEmbeddings("mix-backward", chars_per_chunk=64, fine_tune=fine_tune_embeddings)]
        elif embedding == "bert":
            print("[flair] initializing Bert embeddings", flush=True)
            embeddings_list += [TransformerWordEmbeddings('bert-base-uncased', layers='-1', fine_tune=fine_tune_embeddings)]
        elif embedding == "elmo":
            print("[flair] initializing ELMo embeddings", flush=True)
            embeddings_list += [ELMoEmbeddings(model="medium", embedding_mode="top")]
        else:
            raise NotImplementedError("Embeddings must be in ['flair', 'bert', 'elmo']")

        return StackedEmbeddings(embeddings=embeddings_list).to(device)
    
    elif embedding in ["bert-base", "bert-mix", "bert-last-four"]:
        return None

    else:
        raise NotImplementedError("Unsupported embedding: " + embedding)


def prepare_embeddings_flair(sentences, embedder, device, params):
    embedder.embed(sentences)

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
    labels = torch.as_tensor(np.array([int(s.labels[0].value) for s in sentences]))
    return embeddings.to(device), labels.to(device)


def prepare_embeddings_bert_mix(data, embedder, device, params):
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))
    labels = labels.to(device)

    num_combined_per_gru = int(12 / params.model.num_grus)

    h = [torch.cat(embeddings[2][i*num_combined_per_gru+1 : (i+1)*num_combined_per_gru+1], 2) for i in range(params.model.num_grus)]
    return h, labels
    # h0 = torch.cat(embeddings[2][1:5], 2)
    # h1 = torch.cat(embeddings[2][5:9], 2)
    # h2 = torch.cat(embeddings[2][9:13], 2)
    # return [h0, h1, h2], labels


def prepare_embeddings_bert_base(data, embedder, device, params):
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))
    labels = labels.to(device)
    # h2 = torch.cat(embeddings[2][12], 2)
    h2 = embeddings[2][12]
    return [h2], labels


def prepare_embeddings_bert_last_four(data, embedder, device, params):
    x = data["text"]
    labels = data["label"]
    embeddings = embedder(input_ids=x.to(device))
    labels = labels.to(device)
    h2 = torch.cat(embeddings[2][9:13], 2)
    return [h2], labels