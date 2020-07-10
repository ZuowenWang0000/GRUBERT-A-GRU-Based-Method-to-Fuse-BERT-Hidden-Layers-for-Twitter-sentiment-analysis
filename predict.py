import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from attention_network import AttentionNetwork
# import tensorflow.compat.v1 as tf
# import tensorflow_hub as hub
from dataset import TweetsDataset
from utils import *
from load_embeddings import *
from torch.utils.tensorboard import SummaryWriter
import json
import click
import os
import copy
import numpy as np
import pandas as pd

def predict(eval_loader, model, device, config, elmo):
    model.eval()
    length = config.model.sentence_length_cut
    results = np.array([])
    for i, (data, tweet) in enumerate(eval_loader):
        embeddings = data["embeddings"]
        embeddings = embeddings.to(device)
        labels = data["label"]
        elmo_embeddings = torch.Tensor(elmo.embed(np.array(tweet).T, [length for _ in range(len(labels))])).to(device)
        # print(elmo_embeddings.shape)
        embeddings = torch.cat([embeddings, elmo_embeddings], 2)

        # Forward prop.
        scores, word_alphas, emb_weights = model(embeddings)

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)

        results = np.concatenate((results, predictions.cpu().numpy()))
        print(i)
    return results

def predict_bert_mix(eval_loader, model, device, config, embedder):
    model.eval()  # training mode enables dropout
    results = np.array([])
    # Batches
    for i, data in enumerate(eval_loader):
        # embeddings = torch.tensor(data["embeddings"])
        x = data["text"]
        embeddings = model.model(input_ids=x.to(device))

        h0 = torch.cat(embeddings[2][1:5], 2)
        h1 = torch.cat(embeddings[2][5:9], 2)
        h2 = torch.cat(embeddings[2][9:13], 2)

        # Forward prop.
        scores, word_alphas, emb_weights = model([h0, h1, h2])

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        results = np.concatenate((results, predictions.cpu().numpy()))
        print(i)
    return results


def predict_flair(eval_loader, model, device, config, embedder):
    model.eval()  # training mode enables dropout
    length = config.model.sentence_length_cut
    results = np.array([])
    for i, sentences in enumerate(eval_loader):
        # batch_start = time.time()
        # embeddings = torch.tensor(data["embeddings"])
        # Perform embedding + padding
        embedder.embed(sentences)

        lengths = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            embedder.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float, device=device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[ : embedder.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        embeddings = torch.cat(all_embs).view([
                len(sentences),
                longest_token_sequence_in_batch,
                embedder.embedding_length,
            ]
        )

        embeddings = embeddings.to(device)
        scores, word_alphas, emb_weights = model(embeddings)
        
        for sentence in sentences:
            sentence.clear_embeddings()

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        results = np.concatenate((results, predictions.cpu().numpy()))
        print(i)
    return results


@click.command()
@click.option('--config', default='configs/pipeline_check_lstm.json', type=str)
@click.option('--checkpoint', default='./log_dir/')
@click.option('--predict-file', default='./prediction', type=str)
@click.option('--embedding', default='elmo', type=str)
# @click.option('--use-flair', default=False, type=bool)
# @click.option('--use-bert', default=False, type=bool)


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
    # device = "cpu"
    # setup embeddings
    
    if embedding=="flair":
        from flair.embeddings import WordEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
        print("[flair] initializing embeddings", flush=True)
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        flair_forward_embedding = FlairEmbeddings("mix-forward", chars_per_chunk=64)
        flair_backward_embedding = FlairEmbeddings("mix-backward", chars_per_chunk=64)
        embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, flair_forward_embedding, flair_backward_embedding])

        import flair
        from flair.datasets import CSVClassificationDataset
        print("[flair] initializing datasets", flush=True)
        eval_dataset = CSVClassificationDataset(os.path.join(dataset_path, test_file_path), {1: "text", 2: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        eval_loader = flair.datasets.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        embedder = embedding.to(device)

        prediction_func = predict_flair
        print("using bert embedding for testing")

    elif embedding=="bert":
        from flair.embeddings import WordEmbeddings, ELMoEmbeddings,  TransformerWordEmbeddings, StackedEmbeddings
        print("[flair] initializing embeddings", flush=True)
        glove_embedding = WordEmbeddings("../embeddings/glove.6B.300d.gensim")
        syngcn_embedding = WordEmbeddings("../embeddings/syngcn.gensim")
        bert_embedding = TransformerWordEmbeddings('bert-base-uncased', layers='-1')
        embedding = StackedEmbeddings(embeddings=[glove_embedding, syngcn_embedding, bert_embedding])

        import flair
        from flair.datasets import CSVClassificationDataset
        print("[flair] initializing datasets", flush=True)
        eval_dataset = CSVClassificationDataset(os.path.join(dataset_path, test_file_path), {1: "text", 2: "label"}, max_tokens_per_doc=sentence_length_cut, tokenizer=False, in_memory=False, skip_header=True)
        eval_loader = flair.datasets.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        embedder = embedding.to(device)

        prediction_func = predict_flair
        print("using bert embedding for testing")

    elif embedding=="elmo":
        import tensorflow as tf
        import tensorflow_hub as hub
        glove_embedding = GloveEmbedding(dataset_path, train_file_path, test_file_path, sentence_length_cut)
        syngcn_embedding = SynGcnEmbedding(dataset_path, train_file_path, test_file_path, sentence_length_cut, "../embeddings/syngcn_embeddings.txt")

        # dataloader
        eval_loader = torch.utils.data.DataLoader(TweetsDataset(glove_embedding.get_test_set(), syngcn_embedding.get_test_set()),
                                                batch_size=100, shuffle=False,
                                                num_workers=workers, pin_memory=True)

        # initialzie elmo
        # sess = tf.Session()
        elmo = hub.load("https://tfhub.dev/google/elmo/3")
        # sess.run(tf.global_variables_initializer())
        embedder = ElmoEmbedding(elmo, None)
        prediction_func = predict
    
    elif embedding=="bert-mix":
        from attention_network import AttentionNetwork
        from lstm_model import LstmModel
        from gru_model import GruModel
        from bert_model import BertSentimentModel
        from dataset import BertTwitterDataset
        print("[bert-mix] initializing embeddings+dataset", flush=True)
        # train_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, train_file_path))
        test_dataset = BertTwitterDataset(csv_file=os.path.join(dataset_path, test_file_path))
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)  # should shuffle really be false? copying from the notebook
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        embedder = None

        prediction_func = predict_bert_mix
        print("[bert-mix] entering training loop", flush=True)


    else:
        raise NotImplementedError

    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']

    results = prediction_func(eval_loader, model, device, config, embedder)
    results = ((results-0.5)*2)
    sub = pd.read_csv("./sample_submission.csv", index_col=False)
    sub["Prediction"] = results.astype(int)
    sub.to_csv(predict_file, index=False)


if __name__ == '__main__':
    main_cli()
