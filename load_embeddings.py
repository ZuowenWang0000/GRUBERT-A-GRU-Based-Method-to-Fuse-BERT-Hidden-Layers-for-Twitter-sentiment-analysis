import torchtext.vocab as vocab
from torchtext.data import Field
from torchtext.data import TabularDataset
import os

def tokenize(x):
    return x.split()

def get_glove_embedding(data_path):
    text_field_glove = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=40)
    label_field_glove = Field(sequential=False, use_vocab=False)

    data_fields_glove = [("text", text_field_glove), ("label", label_field_glove)]

    train_glove, valid_glove = TabularDataset.splits(
                path=data_path,
                train='train_small_split.csv', validation="val_small_split.csv",
                format='csv',
                skip_header=True,
                fields=data_fields_glove)
    glove_type = 'glove.6B.300d'
    print("**************USING GLOVE TYPE: {} ****************".format(glove_type))
    text_field_glove.build_vocab(train_glove, vectors=glove_type)
    return text_field_glove.vocab
    # return text_field_glove.vocab.vectors

def get_syngcn_embedding(data_path, syngcn_path):
    text_field_synGCN = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length = 40)
    label_field_synGCN = Field(sequential=False, use_vocab=False)

    data_fields_synGCN = [("text", text_field_synGCN), ("label", label_field_synGCN)]

    train_synGCN, valid_synGCN = TabularDataset.splits(
                path=os.path.join(data_path, "torchtext_data"),
                train='train_small_split.csv', validation="val_small_split.csv",
                format='csv',
                skip_header=True,
                fields=data_fields_synGCN)
    
    syngcn = vocab.Vectors(name=os.path.join(syngcn_path, "embeddings/syngcn_embeddings.txt"))
    text_field_synGCN.build_vocab(train_synGCN, vectors=syngcn)
    return text_field_synGCN.vocab.vectors