# GRUBERT: A GRU-Based Method to Fuse BERT Hidden Layers for Twitter sentiment analysis

<img align="right" width="180" src="https://github.com/ZuowenWang0000/GRUBERT-A-GRU-Based-Method-to-Fuse-BERT-Hidden-Layers-for-Twitter-sentiment-analysis/blob/master/grubert.png">

If you use our code please consider citing
```
@inproceedings{horne-etal-2020-grubert,
    title = "{GRUBERT}: A {GRU}-Based Method to Fuse {BERT} Hidden Layers for {T}witter Sentiment Analysis",
    author = "Horne, Leo  and
      Matti, Matthias  and
      Pourjafar, Pouya  and
      Wang, Zuowen",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing: Student Research Workshop",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.aacl-srw.19",
    pages = "130--138"
}
```
# Table of Contents
- [Table of Contents](#table-of-contents)
- [Preparations](#preparations)
  - [Dependencies](#dependencies)
  - [Dataset](#dataset)
- [Training](#training)
  - [Available models and embeddings](#available-models-and-embeddings) 
    - [Compatible transformers-based embeddings](#compatible-transformers-based-embeddings)
    - [Compatible Flair-based embeddings](#compatible-flair-based-embeddings)
      - [Additional embedding files required for GloVe and SynGCN](#additional-embedding-files-required-for-glove-and-syngcn)
- [Prediction](#prediction)
- [Majority Voting](#majority-voting)
- [Configuration Files](#configuration-files)

# Preparations

## Dependencies 
For Leonhard users, to install dependencies, please first execute
```
module load eth_proxy
module load gcc/6.3.0 python_gpu/3.7.4 hdf5
```
Then one can use pip to install
```
pip3 install --user -r requirements.txt
```
to install the dependencies needed. Virtual environment is recommended.

## Dataset
Download the preprocessed tweet datasets from [here](https://polybox.ethz.ch/index.php/s/Tb0QWEKEK9Bhiqy?path=%2Fdataset%2Ffinal_dataset).

This link contains a train split (train_split.csv, 80% of cleaned trianing set), a validation split (val_split.csv, the rest 20%) and a test split (test_cleaned.csv, label unknown). 
The datapath is controlled by configuration files in the config folder.

The train and validation splits are created by spliting the original training set provided by the ETH CIL course team.
These datasets are preprocessed using the same preprocessing procedure described in the report Section 2.1.

The scripts for preprocessing can be found in the `preprocessing` directory. `preprocess.py` performs spell-checking, emoji, <user>, and <url> replacement, duplicate tweet removal, and extraneous whitespace removal. Then, `preprocess2.py` removes tweets that consist solely of whitespace.

# Training
For Leonhard users please execute train.sh with flags:
```
./train.sh --config configs/subfolder/the_experiment_config_file.json 
           --embedding <embedding> (see next subsection) --seed 0
```

For non-Leonhard users please execute the python train.py script. Flags are the same as on Leonhard and can be viewed using `python train.py --help`


an example for a quick start (conrrespond to bert-share-3 in the report Table 2) on Leonhard:
```
./train.sh --config configs/bert_share_configs/bert_share_3.json 
           --embedding bert-mix --seed 0
```

## Available models and embeddings
There are two classes of models: 
1. models that operate on bert-based (or RoBERTa-based) embeddings from huggingface's transformers library, and 
2. models that operate on embeddings provided by the Flair NLP library. 

In general, flair-based models (GSFlairMixModel, LstmModel) can operate with any compatible flair-based embedding (see list below), and bert-based models (BertMixModel, BertWSModel, BertLinearModel, BertMixLSTMModel) can operate on any compatible transformers-based embedding (see list below)

### Compatible transformers-based embeddings
- `bert-base`: Uses the last BERT hidden layer
- `bert-last-four`: Uses a concatenation of the last 4 BERT hidden layers
- `bert-mix`: Uses BERT embeddings, exposing all 12 hidden layers
- `roberta-mix`: Uses Roberta embeddings, exposing all 12 hidden layers


### Compatible Flair-based embeddings
GloVe and SynGCN embeddings require additional files to be present in the `./embeddings` directory (see the next section)

- `flair`: Uses Flair forward and backward embeddings 
- `gs-flair`: Uses a mix of GloVe, SynGCN, and Flair forward and backward embeddings 
- `elmo`: Uses ELMo embeddings
- `gs-elmo`: Uses a mix of GloVe, SynGCN, and ELMo embeddings
- `gs-bert`: Uses a mix of GloVe, SynGCN, and BERT embeddings
- `glove`: Uses GloVe embeddings
- `syngcn`: Uses SynGCN embeddings
- `gs-only`: Uses a mix of GloVe and SynGCN embeddings
- `twitter`: Uses Twitter embeddings from Flair

#### Additional embedding files required for GloVe and SynGCN

These are available for download [here](https://polybox.ethz.ch/index.php/s/Tb0QWEKEK9Bhiqy?path=%2Fembeddings) and should be placed into the `./embeddings` directory.


# Prediction
We save a checkpoint after every epoch. For making prediction on the test dataset, one needs to run the 
predict.sh script and specify the config file, checkpoint path and the file name where the predictions are stored. An example is as follows:
```
./predict.sh --config config/the_experiment_config_file.json 
             --checkpoint_path /cluster/scratch/nethz/logdir/bert_share_3_bs64_ft_para_seed0/checkpoint_2.tar 
             --predict-file ./pred_share_3_ep2_s0.csv
```

# Majority Voting
The code for majority voting is in `maj_vote.py`. The file must be manually modified with the names of the prediction files to do majority voting on.

# Configuration Files
A typical configuration file to control the model type, model parameter and experiment environment looks as follows:

```
{
    "model": {                                                      #NOTE: model parameters differ according to the model;
                                                                                 this is an example for a BertMixModel config
        "architecture": "BertMixModel",     #other options: see previous section
        "n_classes": 2,                                        #number of classes for prediction (here, just positive and negative)
        "gru_hidden_size": 100,                     #the number of hidden units in each GRU used in the model
        "num_gru_layers": 1,                          #the number of layer in each GRU used in the model.
        "num_grus": 3,                                       #the number of GRUs used to fuse the bert layers. 
                                                                                Refer to Section 2.2 for more details.
        "linear_hidden_size": 100,                #the number of hidden units for the linear classifier layer 
        "dropout": 0.5, 
        "fine_tune_embeddings": true,      #to reveal the true power of bert, fine-tune need to be enabled
        "sentence_length_cut": 40, 
        "use_regularization": "none",          #used in pre-study experiments, not in final report
        "regularization_lambda": 0              #used in pre-study experiments, not in final report
    },
    "training": {
        "start_epoch": 0,                                   #the starting epoch, only used for continue training, otw set to 0
        "batch_size": 64,  
        "lr": 1e-5,
        "lr_decay": 0.9,                                       #the learning rate decays to previous learning rate * lr_decay in each epoch
        "momentum": 0.9,
        "workers": 0, 
        "epochs": 30,
        "grad_clip": "none",
        "print_freq": 250,
        "checkpoint": "none",
        "save_checkpoint_freq_epoch": 1,  
        "save_checkpoint_path": "/cluster/scratch/__USER__/logdir/bert_share_3_bs64_ft_para",  
         # specify the path to save the checkpoint 
         # -- NOTE: if training on a local machine instead of Leonhard, the checkpoint path will need to be changed
        "train_without_val": false,
        "weight_decay":0.0
    },
    "dataset": {
        "dataset_dir": "../dataset",               #the dataset folder (which includes train, validation and test files)
        "rel_train_path": "train_split.csv",
        "rel_val_path": "val_split.csv",
        "rel_test_path": "test_cleaned.csv"
    }
}

```




