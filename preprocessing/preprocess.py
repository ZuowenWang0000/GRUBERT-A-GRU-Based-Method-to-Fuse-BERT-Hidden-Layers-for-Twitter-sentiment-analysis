import re
from fastai import *
from fastai.text import *
from autocorrect import Speller

start_index = int(input("start index\n"))
end_index = int(input("end inex\n"))

tweets = list(set(open("../dataset/train_pos_full.txt", "r").readlines()))
print("length of cleaned tweets {}".format(len(tweets)))
spell = Speller()
fp = open("../dataset/train_pos_{}_{}.txt".format(start_index, end_index), "w")

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
    if(i%1000 == 0):
        print("iter {}".format(i))
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
