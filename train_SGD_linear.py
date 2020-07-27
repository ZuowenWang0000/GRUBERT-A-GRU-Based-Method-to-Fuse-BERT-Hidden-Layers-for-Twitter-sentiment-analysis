import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gensim
from gensim import utils
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import sys
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import multiprocessing as mp
import pandas

def avg_text_embedding(tokenized_text, embeddings_dict, embed_dim, vocab):
# def avg_text_embedding(embeddings_dict, embed_dim, vocab):
    # def avg_text_embedding_funtional(tokenized_text):
        length = len(tokenized_text)
        if length == 0:
            # print("length = {}".format(length))
            # print(tokenized_text)
            return np.zeros(embed_dim)
        sum = np.zeros(embed_dim)
        for i in range(length):
            if(tokenized_text[i] in vocab): #skip if it's out of vocabulary
                sum += np.array(embeddings_dict[tokenized_text[i]])
        return sum/length
    # return avg_text_embedding_funtional

def build_avg_text_bedding_dataset(train_preprocessed_text, embeddings_dict, embed_dim, vocab):
    dataset_size = len(train_preprocessed_text)
    feature_np = np.zeros(shape=(dataset_size, embed_dim))

    # try:
    #     cpus = mp.cpu_count()-2 # leave me 2 cores to browse youtube
    # except NotImplementedError:
    #     cpus = 2  # arbitrary default

    for i in range(dataset_size):
        feature_np[i] = avg_text_embedding(train_preprocessed_text[i], embeddings_dict, embed_dim, vocab)

    # pool = mp.Pool(processes=cpus)
    # f = avg_text_embedding(embeddings_dict, embed_dim, vocab)
    # feature_np = pool.map(f, train_preprocessed_text)
    return feature_np



def build_tokenized_dataset(raw_text, stop_words):
    dataset_size = len(raw_text)
    tokenized_dataset = []

    for i in range(dataset_size):
        tokenized_dataset.append(preprocess(raw_text[i], stop_words))
    return tokenized_dataset

def main():
    print("running main")
    """
       load raw text, make dataset
    """

    try:
        features_np = np.load('./dataset/features_np.npy')
        labels = np.load('./temp_data/labels.npy')
    except FileNotFoundError:
        print("file not found, regenerating")
        download('punkt')  # tokenizer, run once
        download('stopwords')  # stopwords dictionary, run once
        stop_words = stopwords.words('english')

        pos_text, neg_text, vocab = load_train(train_on_full=True)
        train = pos_text + neg_text
        labels = make_labels(len(pos_text), len(neg_text))

        assert len(train) == len(labels)

        embeddings_dict = load_embedding(vocab, "./embedding/std_glove_embeddings.npz")
        embed_dim = len(embeddings_dict['good'])

        # tokenize and remove stop_words
        tokenized_train = build_tokenized_dataset(train, stop_words)

        # get the averaged embedding
        features_np = build_avg_text_bedding_dataset(tokenized_train, embeddings_dict, embed_dim, vocab)
        np.save('./temp_data/features_np', features_np)
        np.save('./temp_data/labels', labels)

    try:
        test_features_np = np.load('./temp_data/test_features_np.npy')
    except FileNotFoundError:
        print("file not found, regenerating")
        download('punkt')  # tokenizer, run once
        download('stopwords')  # stopwords dictionary, run once
        stop_words = stopwords.words('english')

        test_text, vocab = load_test()

        embeddings_dict = load_embedding(vocab, "./embedding/std_glove_embeddings.npz")
        embed_dim = len(embeddings_dict['good'])

        # tokenize and remove stop_words
        tokenized_test = build_tokenized_dataset(test_text, stop_words)

        # get the averaged embedding
        test_features_np = build_avg_text_bedding_dataset(tokenized_test, embeddings_dict, embed_dim, vocab)
        np.save('./temp_data/test_features_np', test_features_np)


    X_train, X_val, y_train, y_val  = train_test_split(
        features_np, labels,
        train_size=0.8, shuffle = True, random_state=1234)

    print(X_train.shape)
    print(y_train.shape)

    model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                          max_iter=1000, tol=0.0001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1,
                          random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False,
                          validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("validation set accuracy: {}".format(accuracy_score(y_val, clf.predict(X_val))))


#   predict
    df = pandas.read_csv('sample_submission.csv')

    print(clf.predict(test_features_np).size)

    df['Prediction'] = clf.predict(test_features_np).astype('int')
    df.to_csv('prediction',index=False)


if __name__ == '__main__':
    main()