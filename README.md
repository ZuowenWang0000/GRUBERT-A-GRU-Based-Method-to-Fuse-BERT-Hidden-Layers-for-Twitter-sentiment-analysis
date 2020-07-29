
## Big files of our CIL project
https://polybox.ethz.ch/index.php/s/Tb0QWEKEK9Bhiqy
=======
# Table of Contents
- [Table of Contents](#table-of-contents)
- [Preparations](#preparations)
  - [Dependencies](#dependencies)
  - [Dataset](#dataset)
- [Training](#training)
  - [Available models and embeddings](#available-models-and-embeddings)
    - [Compatible Flair-based embeddings](#compatible-flair-based-embeddings)
      - [Additional embedding files required for GloVe and SynGCN](#additional-embedding-files-required-for-glove-and-syngcn)
    - [Compatible transformers-based embeddings](#compatible-transformers-based-embeddings)
- [Prediction](#prediction)
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

<<<<<<< HEAD
Download the tweet datasets from here:
https://polybox.ethz.ch/index.php/s/pp6Mzg7BwcXVG5z
=======
## Dataset
Download the tweet datasets from [here](https://polybox.ethz.ch/index.php/s/Tb0QWEKEK9Bhiqy?path=%2Fdataset%2Ffinal_dataset).


This link contains a train split, a validation split and a test split (label unknown). The train and validation splits are created by spliting the original training set provided by the ETH CIL course team.
These datasets are preprocessed using the same preprocessing procedure described in the report Section 2.1.

# Training
For Leonhard users please execute train.sh with flags:
```
./train.sh --config configs/the_experiment_config_file.json --embedding <embedding>
```

<<<<<<< HEAD
- vocab.txt vocab.pkl vocab_cut.txt
- cooc.pkl: cooccurance matrix 

## Build the Co-occurence Matrix (already in datasets but feel free to rerun)
=======
For non-Leonhard users please execute the python train.py script. Flags are the same as on Leonhard and can be viewed using `python train.py --help`
>>>>>>> c326bebd1b69cd16f0b8ca8fd1c57140217ebe62


an example for a quick start on Leonhard:
```
./train.sh --config configs/bert_mix.json --embedding bert_mix --seed 0
```

## Available models and embeddings
There are two classes of models: models that operate on bert-based embeddings from huggingface's transformers library, and models that operate on embeddings provided by the Flair NLP library. In general, flair-based models (GSFlairMixModel, LstmModel) can operate with any compatible flair-based embedding (see list below), and bert-based models (BertMixModel, BertWSModel, BertLinearModel, BertMixLSTMModel) can operate on any compatible transformers-based embedding (see list below)

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

### Compatible transformers-based embeddings
- `bert-base`: Uses the last BERT hidden layer
- `bert-last-four`: Uses a concatenation of the last 4 BERT hidden layers
- `bert-mix`: Uses BERT embeddings, exposing all 12 hidden layers
- `roberta-mix`: Uses Roberta embeddings, exposing all 12 hidden layers

# Prediction
We save a checkpoint every epoch. For making prediction on the test dataset, one needs to run the 
predict.sh script and specify the config file, checkpoint path and the file name where the predictions are stored. An example is as follows:
```
./predict.sh --config config/the_experiment_config_file.json --checkpoint_path /cluster/scratch/hoffmannthebestman/log_dir/bert_mix_seed0/checkpoint_han_2.tar --predict-file ./pred_bert_mix_s0.csv
```

# Configuration Files
A typical configuration file to control the model type, model parameter and experiment environment looks as follows:
```
{
    "model": {  # NOTE: model parameters differ according to the model; this is an example for a BertMixModel config
        "architecture": "BertMixModel",   # other options: see previous section
        "n_classes": 2,  # number of classes for prediction (here, just positive and negative)
        "gru_hidden_size": 100,  # the number of hidden units in each GRU used in the model, plz refer to section XX in the paper
        "num_gru_layers": 1, # the number of layer in each GRU used in the model.
        "num_grus": 3, # the number of GRUs used to fuse the bert layers
        "linear_hidden_size": 100, # the number of hidden units for the linear classifier layer 
        "dropout": 0.5, 
        "fine_tune_embeddings": true, # to reveal the true power of bert, fine-tune need to be enabled
        "sentence_length_cut": 40, 
        "use_regularization": "none",  # parameters used in the experiment in appendix XX
        "regularization_lambda": 0  # for appendix XX
    },
    "training": {
        "start_epoch": 0,  # the starting epoch, only used for continue training, otw set to 0
        "batch_size": 64,  
        "lr": 1e-5,
        "lr_decay": 0.9, # after each iteration, the learning rate will be set to previous learning rate * lr_decay
        "momentum": 0.9,
        "workers": 0, 
        "epochs": 30,
        "grad_clip": "none",
        "print_freq": 250,
        "checkpoint": "none",
        "save_checkpoint_freq_epoch": 1,  
        "save_checkpoint_path": "/cluster/scratch/__USER__/log_dir/mix_bert_bs64",  # specify the path to save the checkpoint 
            # -- NOTE: if training on a local machine instead of Leonhard, the checkpoint path will need to be changed
        "train_without_val": false,
        "weight_decay":0.0
    },
    "dataset": {
        "dataset_dir": "../dataset",  # the dataset folder (which includes train, validation and test files)
        "rel_train_path": "train_split.csv",
        "rel_val_path": "val_split.csv",
        "rel_test_path": "test_cleaned.csv"
    }
}

```


<<<<<<< HEAD
##  Template for Glove Question (already in polybox embeddings)
=======



<<<<<<< HEAD
Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets train_pos_full.txt, train_neg_full.txt

Note: std_glove_embeddings.npz is trained with the glove_solution.py provided by the course group. We should use it in part of the baseline.
can be downloaded from:
https://polybox.ethz.ch/index.php/s/JQ8awPuk5tdrp5A


##  available embedding so far
standard glove embedding (provided by the TA group)
https://polybox.ethz.ch/index.php/s/JQ8awPuk5tdrp5A
=======

