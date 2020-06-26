import torchtext.vocab as vocab
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
import tensorflow_hub as hub
import tensorflow as tf
tf.disable_eager_execution()
import os

def tokenize(x):
    return x.split()

class GloveEmbedding:
    def __init__(self, data_path, train_csv_file, test_csv_file, sentence_length_cut):
        self.data_path = data_path
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.sentence_length_cut = sentence_length_cut
        self.initialize_embeddings()

    def initialize_embeddings(self):
        self.text_field_glove = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=self.sentence_length_cut)
        label_field_glove = Field(sequential=False, use_vocab=False)

        data_fields_glove = [("text", self.text_field_glove), ("label", label_field_glove)]

        self.train_glove, self.valid_glove = TabularDataset.splits(
                    path=self.data_path,
                    train=self.train_csv_file, validation=self.test_csv_file,
                    format='csv',
                    skip_header=True,
                    fields=data_fields_glove)
        glove_type = 'glove.6B.300d'
        print("**************USING GLOVE TYPE: {} ****************".format(glove_type))
        self.text_field_glove.build_vocab(self.train_glove, vectors=glove_type)
        # train_iter, valid_iter = BucketIterator.splits(
        #     (self.train_glove, self.valid_glove), 
        #     batch_sizes=(1,1), 
        #     device=None,
        #     sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
        #     sort_within_batch=True,
        # )
        # print(self.train_glove.fields["text"].process([self.train_glove[0].text]))
        # print(self.train_glove[0])
        # # print([x.text[0] for x in train_iter])
        # print(next(iter(train_iter)).text)
        # print(self.text_field_glove.vocab.itos[1])
        # temp = self.text_field_glove.vocab.vectors[self.train_glove.fields["text"].process([self.train_glove[0].text])[0].T]
        # temp = temp.squeeze(0)
        # print(temp)
        # print(temp.shape)
        # print(self.train_glove[0].label)
        # to_return = train_glove if train_or_test == "train" else valid_glove
        # return text_field_glove, to_return 

    def get_train_set(self):
        return self.text_field_glove, self.train_glove

    def get_test_set(self):
        return self.text_field_glove, self.valid_glove

class SynGcnEmbedding:
    def __init__(self, data_path, train_csv_file, test_csv_file, sentence_length_cut, syngcn_path):
        self.data_path = data_path
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.sentence_length_cut = sentence_length_cut
        self.syngcn_path = syngcn_path
        self.initialize_embeddings()

    def initialize_embeddings(self):

        self.text_field_synGCN = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=self.sentence_length_cut)
        label_field_synGCN = Field(sequential=False, use_vocab=False)

        data_fields_synGCN = [("text", self.text_field_synGCN), ("label", label_field_synGCN)]

        self.train_synGCN, self.valid_synGCN = TabularDataset.splits(
                    path=self.data_path,
                    train=self.train_csv_file, validation=self.test_csv_file,
                    format='csv',
                    skip_header=True,
                    fields=data_fields_synGCN)
        
        syngcn = vocab.Vectors(name=self.syngcn_path)
        self.text_field_synGCN.build_vocab(self.train_synGCN, vectors=syngcn)

    def get_train_set(self):
        return self.text_field_synGCN, self.train_synGCN

    def get_test_set(self):
        return self.text_field_synGCN, self.valid_synGCN

class ElmoEmbedding:
    def __init__(self, elmo, sess):
        # self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
        self.elmo = elmo
        self.sess = sess
    def embed(self, sentence, length):
        # sentence= ["dog", "hello"]
        # print(sentence)
        # print(len(sentence))
        return self.sess.run(self.elmo(inputs={"tokens": sentence, "sequence_len": length}, signature="tokens", as_dict=True)["word_emb"])

# def get_syngcn_embedding(data_path, syngcn_path):
#     text_field_synGCN = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length = 40)
#     label_field_synGCN = Field(sequential=False, use_vocab=False)

#     data_fields_synGCN = [("text", text_field_synGCN), ("label", label_field_synGCN)]

#     train_synGCN, valid_synGCN = TabularDataset.splits(
#                 path=os.path.join(data_path, "torchtext_data"),
#                 train='train_small_split.csv', validation="val_small_split.csv",
#                 format='csv',
#                 skip_header=True,
#                 fields=data_fields_synGCN)
    
#     syngcn = vocab.Vectors(name=os.path.join(syngcn_path, "embeddings/syngcn_embeddings.txt"))
#     text_field_synGCN.build_vocab(train_synGCN, vectors=syngcn)
#     return text_field_synGCN.vocab.vectors
