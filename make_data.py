import pandas as pd
from collections import namedtuple
import os
from sklearn.model_selection import train_test_split
import numpy
import sys

def make_training_validation_split(path):

    train_df = pd.read_csv(os.path.join(path, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(path, "val_split.csv"))

    main_df = train_df.append(val_df)
    train , val = train_test_split(main_df,stratify=main_df['label'],test_size=0.3,shuffle=True)

    train.to_csv(os.path.join(path, "train_split_new.csv"), header=True,index=False)
    val.to_csv(os.path.join(path, "val_split_new.csv"),header=True,index=False)

if __name__ == '__main__':
    make_training_validation_split(sys.argv[1])
