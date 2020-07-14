import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils import *
from dataset import BertTwitterDataset
from torch.utils.tensorboard import SummaryWriter
import json
import click
import os
import copy
import numpy as np
import pandas as pd

def prepare_embeddings_flair(sentences, embedder, device):
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

    return embeddings.to(device)


def prepare_embeddings_bert(data, embedder, device):
    x = data["text"]
    embeddings = embedder(input_ids=x.to(device))

    h0 = torch.cat(embeddings[2][1:5], 2)
    h1 = torch.cat(embeddings[2][5:9], 2)
    h2 = torch.cat(embeddings[2][9:13], 2)

    return [h0, h1, h2]


def predict(eval_loader, model, device, config, prepare_embeddings_fn, embedder):
    model.eval()  # eval mode disables dropout
    results = np.array([])
    # Batches
    for i, data in enumerate(eval_loader):
        embeddings = prepare_embeddings_fn(data, embedder, device)

        # Forward prop.
        output = model(embeddings)

        # Find accuracy
        _, predictions = output["logits"].max(dim=1)  # (n_documents)
        results = np.concatenate((results, predictions.cpu().numpy()))
        print(i)

        try:
            for sentence in data:
                sentence.clear_embeddings()
        except:
            pass
    return results


@click.command()
@click.option('--config', default='configs/pipeline_check_lstm.json', type=str)
@click.option('--checkpoint', default='./log_dir/')
@click.option('--predict-file', default='./prediction', type=str)
@click.option('--embedding', default='elmo', type=str)


def main_cli(config, checkpoint, predict_file, embedding):
    # Dataset parameters
    config_dict = get_config(config)
    config = config_to_namedtuple(config_dict)

    batch_size = 8
    dataset_path = config.dataset.dataset_dir
    train_file_path = config.dataset.rel_train_path
    test_file_path = config.dataset.rel_test_path
    sentence_length_cut = config.model.sentence_length_cut #set fixed sentence length
    workers = config.training.workers  # number of workers for loading data in the DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup embeddings
    
    if embedding in ["flair", "bert", "elmo"]:
        import flair
        from flair.datasets import CSVClassificationDataset
        from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        embeddings_list = [glove_embedding, syngcn_embedding]

        if embedding == "flair":
            print("[flair] initializing Flair embeddings", flush=True)
            embeddings_list += [FlairEmbeddings("mix-forward", chars_per_chunk=64, fine_tune=fine_tune), FlairEmbeddings("mix-backward", chars_per_chunk=64, fine_tune=fine_tune)]
        elif embedding == "bert":
            print("[flair] initializing Bert embeddings", flush=True)
            embeddings_list += [TransformerWordEmbeddings('bert-base-uncased', layers='-1', fine_tune=fine_tune)]
        elif embedding == "elmo":
            print("[flair] initializing ELMo embeddings", flush=True)
            embeddings_list += [ELMoEmbeddings(model="medium", embedding_mode="top")]
        else:
            raise NotImplementedError("Embeddings must be in ['flair', 'bert', 'elmo']")

        embedding = StackedEmbeddings(embeddings=embeddings_list)
        print("[flair] initializing dataset", flush=True)
        eval_dataset = CSVClassificationDataset(os.path.join(dataset_path, train_file_path), {1: "text", 2: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        eval_loader = flair.datasets.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        embedder = embedding.to(device)
        prepare_embeddings_fn = prepare_embeddings_flair
        print("[flair] entering prediction loop", flush=True)
    
    elif embedding == "bert-mix":
        from transformers import BertModel
        print("[bert-mix] initializing embeddings+dataset", flush=True)
        eval_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path))
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)  # should shuffle really be false? copying from the notebook
        embedder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in embedder.parameters():
            param.requires_grad = True
        embedder = embedder.to(device)
        prepare_embeddings_fn = prepare_embeddings_bert
        print("[bert-mix] entering prediction loop", flush=True)

    else:
        raise NotImplementedError("Unsupported embedding: " + embedding)

    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']

    results = predict(eval_loader, model, device, config, prepare_embeddings_fn, embedder)
    results = ((results-0.5)*2)
    sub = pd.read_csv("./sample_submission.csv", index_col=False)
    sub["Prediction"] = results.astype(int)
    sub.to_csv(predict_file, index=False)


if __name__ == '__main__':
    main_cli()
