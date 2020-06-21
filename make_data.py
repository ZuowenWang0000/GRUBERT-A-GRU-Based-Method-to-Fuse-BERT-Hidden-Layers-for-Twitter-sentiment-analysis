import pandas as pd
from collections import namedtuple
import os
from sklearn.model_selection import train_test_split
import numpy

def make_training_validation_split(path):
    Item = namedtuple('Item','text label')
    items = []

    with open(os.path.join(path, "train_pos.txt")) as f: 
        for line in f:
            l = line.rstrip('\n')
            items.append(Item(l, 1))
    with open(os.path.join(path, "train_neg.txt")) as f: 
        for line in f:
            l = line.rstrip('\n')
            items.append(Item(l, 0))

    df = pd.DataFrame.from_records(items, columns=['text', 'label'])

    print(df)
    df.to_csv(os.path.join(path, "/train_small.csv"), index=False, header=True)

    main_df = pd.read_csv(os.path.join(path, "/train_small.csv"))
    train , val = train_test_split(main_df,stratify=main_df['label'],test_size=0.2,shuffle=True)
    train.to_csv(os.path.join(path, "torchtext_data/train_small_split.csv",header=True,index=False))
    val.to_csv(os.path.join("torchtext_data/val_small_split.csv",header=True,index=False))
