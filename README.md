# Preparations

## Dependencies 
Run 
pip install --user requirment.txt
to install the dependencies needed. Virtual environment is recommended.

## Dataset
Download the tweet datasets from here:
https://polybox.ethz.ch/index.php/apps/files/?dir=/CIL/dataset/final_dataset&fileid=1951904263
This link contains a train split, a validation split and a test split (label unknown). The train and validation splits are created by spliting the original training set provided by the ETH CIL course team.
These datasets are preprocessed using the same preprocessing procedure described in the report (section?).

# Training
## For Leonhard users
./train.sh --config config/the_experiment_config_file.json --embedding [bert_base, bert_mix, bert_last_four] 

For non-Leonhard users please execute the python train.py script. Flags are the same as on Leonhard.


an example for a quick start:
./train.sh --config config/bert_mix.json --embedding bert_mix --seed 0

# Prediction
We save a checkpoint every epoch. For making prediction on the test dataset, one needs to run the 
predict.sh script and specify the config file, checkpoint path and the file name where the predictions are stored. An example is as follow:
./predict.sh --config config/the_experiment_config_file.json --checkpoint_path /cluster/scratch/hoffmannthebestman/log_dir/bert_mix_seed0/checkpoint_han_2.tar --predict-file ./pred_bert_mix_s0.csv


# Configuration (detailed explanation)
A typical configuration file to control the model type, model parameter and experiment environment looks like follow:



