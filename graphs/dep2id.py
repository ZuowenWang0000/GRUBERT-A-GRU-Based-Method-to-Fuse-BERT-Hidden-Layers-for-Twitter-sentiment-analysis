import pickle
import spacy
with open("train_neg_full.txt", 'r') as f_pos:
    deps = f_pos.read().splitlines()
with open("train_pos_full.txt", 'r') as f_pos:
    deps1 = f_pos.read().splitlines()

sentence_list = deps+deps1
sentence_list = list(set(sentence_list))
n = len(sentence_list)
nlp = spacy.load("en")
dep2id = {}
uniqueid = 0
for iter,raw_sentence in enumerate(sentence_list[:100000]):
  parsed = nlp(raw_sentence)
  print("ITERATION {} of {}".format(iter,n))
  for thing in parsed.sents:
    for tok in thing:
        if str(tok.dep_) not in dep2id:
            dep2id[str(tok.dep_)] = uniqueid
            uniqueid+=1



with open('dep2id.txt', 'wb') as f:
    pickle.dump(dep2id, f)
