#!/usr/bin/python
#-*-coding:utf-8-*-
from emoji import demojize
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

tweets=[]
meta_data_list = []

dataset_name = '2018'

f = open('data/tweet_data/train.txt', 'r', encoding="utf8")
lines = f.readlines()
for line in lines:
    temp = line.split("\t")
    if(temp[0]=='ID'):
        continue
    tweets.append(normalizeTweet(temp[1]))
    joy=float(temp[2])
    sadness=float(temp[3])
    anger=float(temp[4])
    fear=float(temp[5])
    label=""
    if((joy>sadness) and (joy>anger) and (joy>fear)):
        label="joy"
    elif((sadness>joy) and (sadness>anger) and (sadness>fear)):
        label="sadness"
    elif((anger>joy) and (anger>sadness) and (anger>fear)):
        label="anger"
    else:
        label="fear"
    meta = temp[0] + '\t' + 'train' + '\t' + label
    meta_data_list.append(meta)
f.close()

f = open('data/tweet_data/dev.txt', 'r', encoding="utf8")
lines = f.readlines()
for line in lines:
    temp = line.split("\t")
    if(temp[0]=='ID'):
        continue
    tweets.append(normalizeTweet(temp[1]))
    joy=float(temp[2])
    sadness=float(temp[3])
    anger=float(temp[4])
    fear=float(temp[5])
    label=""
    if((joy>sadness) and (joy>anger) and (joy>fear)):
        label="joy"
    elif((sadness>joy) and (sadness>anger) and (sadness>fear)):
        label="sadness"
    elif((anger>joy) and (anger>sadness) and (anger>fear)):
        label="anger"
    else:
        label="fear"
    meta = temp[0] + '\t' + 'train' + '\t' + label
    meta_data_list.append(meta)
f.close()

f = open('data/tweet_data/test.txt', 'r', encoding="utf8")
lines = f.readlines()
for line in lines:
    temp = line.split("\t")
    if(temp[0]=='ID'):
        continue
    tweets.append(normalizeTweet(temp[1]))
    joy=float(temp[2])
    sadness=float(temp[3])
    anger=float(temp[4])
    fear=float(temp[5])
    label=""
    if((joy>sadness) and (joy>anger) and (joy>fear)):
        label="joy"
    elif((sadness>joy) and (sadness>anger) and (sadness>fear)):
        label="sadness"
    elif((anger>joy) and (anger>sadness) and (anger>fear)):
        label="anger"
    else:
        label="fear"
    meta = temp[0] + '\t' + 'test' + '\t' + label
    meta_data_list.append(meta)
f.close()

meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w', encoding="utf8")
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(tweets)

f = open('data/corpus/' + dataset_name + '.txt', 'w', encoding="utf8")
f.write(corpus_str)
f.close()

f = open('data/corpus/' + dataset_name + '.clean.txt', 'w', encoding="utf8")
f.write(corpus_str)
f.close()


print("completed")