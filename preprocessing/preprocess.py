import re
from fastai import *
from fastai.text import *
from autocorrect import Speller

tweets = list(set(open("../../dataset/train_pos_full.txt", "r").readlines()))
spell = Speller()
fp = open("../../dataset/train_pos_full2.txt", "w")

emoticons = {
        "=|": "xxemotneutral",
        "=-(": "xxemotfrown",
        "=-)": "xxemotsmile",
        "=:": "xxemotneutral",
        "=/": "xxemotfrown",
        "='(": "xxemotfrown",
        "='[": "xxemotfrown",
        "=(": "xxemotfrown",
        "=)": "xxemotsmile",
        "=[": "xxemotfrown",
        "=]": "xxemotsmile",
        "={": "xxemotfrown",
        "=\\": "xxemotfrown",
        ">=(": "xxemotfrown",
        ">=)": "xxemotsmile",
        ">:|": "xxemotneutral",
        ">:/": "xxemotfrown",
        ">:[": "xxemotfrown",
        ">:": "xxemotfrown",
        "|:": "xxemotneutral",
        ";|": "xxemotneutral",
        ";-}": "xxemotsmile",
        ";:": "xxemotneutral",
        ";/": "xxemotfrown",
        ";'/": "xxemotfrown",
        ";'(": "xxemotfrown",
        ";')": "xxemotsmile",
        ";)": "xxemotsmile",
        ";]": "xxemotsmile",
        ";}": "xxemotsmile",
        ";*{": "xxemotfrown",
        ":|": "xxemotneutral",
        ":-|": "xxemotneutral",
        ":-/": "xxemotfrown",
        ":-[": "xxemotfrown",
        ":-]": "xxemotsmile",
        ":-}": "xxemotsmile",
        ":-": "xxemotneutral",
        ":-\\": "xxemotfrown",
        ":;": "xxemotneutral",
        "::": "xxemotneutral",
        ":/": "xxemotfrown",
        ":'|": "xxemotneutral",
        ":'/": "xxemotfrown",
        ":')": "xxemotsmile",
        ":'{": "xxemotfrown",
        ":'}": "xxemotsmile",
        ":'\\": "xxemotneutral",
        ":(": "xxemotfrown",
        ":)": "xxemotsmile",
        ":]": "xxemotsmile",
        ":[": "xxemotfrown",
        ":{": "xxemotfrown",
        ":}": "xxemotsmile",
        ":": "xxemotneutral",
        ":*{": "xxemotfrown",
        ":\\": "xxemotfrown",
        "(=": "xxemotsmile",
        "(;": "xxemotsmile",
        "(':": "xxemotsmile",
        ")=": "xxemotfrown",
        ")':": "xxemotfrown",
        "[;": "xxemotsmile",
        "]:": "xxemotfrown",
        "{:": "xxemotsmile",
        "\\=": "xxemotfrown",
        "\\:": "xxemotfrown"
}

for i in range(len(tweets)):
    tweets[i] = tweets[i].strip()
    tweets[i] = re.sub("<user>", "xxuser", tweets[i])
    tweets[i] = re.sub("<url>", "xxurl", tweets[i])
    # tweets[i] = spec_add_spaces(rm_useless_spaces(tweets[i]))
    tweets[i] = re.sub("\\s+", " ", tweets[i])
    tweets[i] = tweets[i].replace("#", "# ")
    tweets[i] = spell(tweets[i])
    tweets[i] = tweets[i].lower()
    for key in emoticons:
        tweets[i] = tweets[i].replace(key, emoticons[key])
    fp.write(tweets[i] + "\n")