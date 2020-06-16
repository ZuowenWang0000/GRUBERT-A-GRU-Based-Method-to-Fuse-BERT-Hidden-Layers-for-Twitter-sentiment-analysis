import nltk
import pickle
import spacy



nlp = spacy.load("en")


with open("all_clean_1000.txt", 'r') as f_pos:
    raw = f_pos.read().splitlines()

"""
print(raw[:1])
tokens = nltk.wordpunct_tokenize(raw)
words = [w.lower for w in tokens]
vocab = sorted(set(tokens))
"""
wordfreq= {}
wordid = {}
uniqueid = 0
for tweet in raw:
    doc = nlp(tweet)
    sentences = [sent.string.strip() for sent in doc.sents]
    for sentence in sentences:
        tokens = nlp(sentence)
        for token in tokens:
            t = str(token)
            if t not in wordfreq.keys():
                wordfreq[t] = 1
            else:
                wordfreq[t] += 1
            if t not in wordid.keys():
                wordid[t] = uniqueid
                uniqueid += 1


with open('voc2id.txt', 'wb') as f:
    pickle.dump(wordid, f)

with open('voc2freq.txt', 'wb') as f:
    pickle.dump(wordfreq, f)
