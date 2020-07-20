# Preparations

## Dependencies 
Run 
```
pip install --user requirment.txt
```
to install the dependencies needed. Virtual environment is recommended.

## Dataset
Download the tweet datasets from here:
https://polybox.ethz.ch/index.php/apps/files/?dir=/CIL/dataset/final_dataset&fileid=1951904263
This link contains a train split, a validation split and a test split (label unknown). The train and validation splits are created by spliting the original training set provided by the ETH CIL course team.
These datasets are preprocessed using the same preprocessing procedure described in the report (section?).

# Training
## For Leonhard users
```
./train.sh --config config/the_experiment_config_file.json --embedding [bert_base, bert_mix, bert_last_four] 
```

For non-Leonhard users please execute the python train.py script. Flags are the same as on Leonhard.


an example for a quick start:
```
./train.sh --config config/bert_mix.json --embedding bert_mix --seed 0
```

# Prediction
We save a checkpoint every epoch. For making prediction on the test dataset, one needs to run the 
predict.sh script and specify the config file, checkpoint path and the file name where the predictions are stored. An example is as follow:
```
./predict.sh --config config/the_experiment_config_file.json --checkpoint_path /cluster/scratch/hoffmannthebestman/log_dir/bert_mix_seed0/checkpoint_han_2.tar --predict-file ./pred_bert_mix_s0.csv
```

# Configuration (detailed explanation)
A typical configuration file to control the model type, model parameter and experiment environment looks like follow:
```
{
    "model": {
        "architecture": "BertMixModel",   # other options: BertBaseModel, BertLastFourModel
        "n_classes": 2,
        "gru_hidden_size": 100,  # the number of hidden units in each GRU used in the model, plz refer to section XX in the paper
        "num_gru_layers": 1, # the number of layer in each GRU used in the model.
        "num_grus": 3, # the number of GRUs used to fuse the bert layers
        "linear_hidden_size": 100, # the number of hidden units for the linear classifier layer 
        "dropout": 0.5, 
        "fine_tune_embeddings": true, # to reveal the true power of bert, fine-tune need to be enabled
        "sentence_length_cut": 40, 
        "device": "torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",  
        "use_regularization": "none",  # parameters used in the experiment in appendix XX
        "regularization_lambda": 0 # for appendix XX
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




