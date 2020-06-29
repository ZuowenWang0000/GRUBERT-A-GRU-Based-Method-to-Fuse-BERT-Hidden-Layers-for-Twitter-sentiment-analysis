import re
import sys
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

# start_index = int(input("start index\n"))
# end_index = int(input("end inex\n"))

emoticons = {
        "xxemotneutral": "neutral",
        "xxemotfrown": "frown",
        "xxemotsmile": "smile",
        "xxemotheart": "heart"
}
file_to_process = str(sys.argv[1])
data = pd.read_csv(file_to_process)
# print(data)
tweets = data['text']

def process2(tweets):
    for i in range(len(tweets)):
        if(i%100 == 0):
            print("iter {}".format(i))
        # tweets[i] = spell(tweets[i])
        tweets[i] = tweets[i].strip()
        tweets[i] = re.sub("xxuser", "user", tweets[i])
        tweets[i] = re.sub("xxurl", "url", tweets[i])

        for key in emoticons:
            tweets[i] = tweets[i].replace(key, emoticons[key])
        if len(tweets)==0:
            continue

def process2_iter(tweet):
    if type(tweet)==float:
        return ""
    tweet = tweet.strip()
    if re.match(r"[^\S\n\t]+", tweet):
        tweet = ""
        return tweet
    tweet = re.sub("xxuser", "user", tweet)
    tweet = re.sub("xxurl", "url", tweet)
    for key in emoticons:
       tweet = tweet.replace(key, emoticons[key])
    return tweet

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs = num_cores)(delayed(process2_iter)(tweet) for tweet in tweets)
print("length before processing :{}".format(len(data['text'])))
data['text'] = results
# drop rows where tweet length == 0
drop_list = []
for i in range(len(data['text'])):
    if len(data['text'][i]) == 0:
        print(data['text'][i])
        drop_list.extend([i])

# print(drop_list)
data = data.drop(drop_list)
# print(data)
print("length after processing :{}".format(len(data['text'])))
data.to_csv(file_to_process + "v2", index=False)
