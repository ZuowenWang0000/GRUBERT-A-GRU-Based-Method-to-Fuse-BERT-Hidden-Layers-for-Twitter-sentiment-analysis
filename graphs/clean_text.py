import fastai
from fastai import *
from fastai.text import *
import spacy
import numpy as np
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
from autocorrect import Speller

import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

train_neg_full_path = "train_neg_full.txt"
train_pos_full_path = "train_pos_full.txt"

with open(train_neg_full_path, 'r') as f_pos:
    neg_text = f_pos.read().splitlines()
with open(train_pos_full_path, 'r') as f_neg:
    pos_text = f_neg.read().splitlines()


pos_text = list(set(pos_text))
neg_text = list(set(neg_text))

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def remove_HTML_spellcheck(sentence_list):
    res = []
    spell = Speller()
    nlp = spacy.load("en")
    tot = len(sentence_list)

    for iter,s in enumerate(sentence_list):
        #print("Original {}".format(s))
        #if iter%100 == 0:
            #print("ITER {} of {}".format(iter,tot))
        s = convert_emoticons(s)
        clean = BeautifulSoup(s, "lxml").text
        right = spell(clean)
        #print("Cleaned sentence {}".format(right))
        res.append(right)
    return res

def clean_sentences(sentence_list):
  fin_list = []
  for s in sentence_list:
    sp = spec_add_spaces(rm_useless_spaces(replace_wrep(replace_rep(fix_html(s)))))
    fin_list.append(sp)
  fin_list = deal_caps(replace_all_caps(fin_list))
  return fin_list





f = clean_sentences(pos_text[:1000])
clean_pos = remove_HTML_spellcheck(f)

with open('pos_clean_1000.txt', 'w') as f:
    for item in clean_pos:
        f.write("%s\n" % item)

clean_neg = remove_HTML_spellcheck(clean_sentences(neg_text[:1000]))


with open('neg_clean_1000.txt', 'w') as f:
    for item in clean_neg:
        f.write("%s\n" % item)


all = clean_neg+clean_pos

with open('all_clean_1000.txt', 'w') as f:
    for item in all:
        f.write("%s\n" % item)
