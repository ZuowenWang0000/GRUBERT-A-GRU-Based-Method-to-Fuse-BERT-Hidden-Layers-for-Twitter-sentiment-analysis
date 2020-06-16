import fastai
from fastai import *
from fastai.text import *
import spacy
import numpy as np
from bs4 import BeautifulSoup
from spellchecker import SpellChecker

train_path = "all_clean_1000.txt"

with open(train_path, 'r') as f_pos:
    alltxt = f_pos.read().splitlines()

with open('voc2id.txt', 'rb') as handle:
  voc2id = pickle.loads(handle.read())

with open('dep2id.txt', 'rb') as handle:
  dep2id = pickle.loads(handle.read())


#mapping from word to unique id
#mapping from deprel to unique id
def make_graphs_coo(sentence_list):

  #returns a list of COO edge indecies

  nlp = spacy.load("en")
  final_list = []
  max_len = 0
  nodes_list = []
  tot_iter = len(sentence_list)
  garbage = ["",","," ", '"','   ']
  for iter,raw_sentence in enumerate(sentence_list):

    #if iter % 5000 == 0:
    #    print ("Iteration {} of {}".format(iter,tot_iter))
    parsed = nlp(raw_sentence)

    rels = []
    for thing in parsed.sents:
      for tok in thing:
            parentid = voc2id[str(tok)]
            childids = [voc2id[str(c)] for c in tok.children]
            print(str(tok.dep_))
            #if str(tok.dep_) in garbage or str(tok.dep_) not in dep2id:
            #    dep = dep2id["NAN"]
            #else:
            dep = dep2id[str(tok.dep_)]
            if len(childids) <= 0:
                continue
            print("{} | {} | {}".format(parentid,childids[0],dep))
            for c in childids:
                rels.append((parentid,c,dep))
    tok_tmp = []
    nwords = 0
    ndeps = 0
    for token in parsed:
        tok_tmp.append(str(voc2id[str(token)]))
        nwords+=1
    for (a,b,d) in rels:
        tok_tmp.append(str(a)+"|"+str(b)+"|"+str(d))
        ndeps+=1
    final_list.append(str(nwords))
    final_list.append(str(ndeps))
    final_list.append(tok_tmp)

  return final_list




all_graphs = make_graphs_coo(alltxt)

with open('graphs.txt', 'w') as f:
    for item in all_graphs:
        f.write("%s\n" % item)
