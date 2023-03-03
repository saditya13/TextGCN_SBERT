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

dataset_name = '2017'

f = open('data/4A-English/SemEval2017-task4-dev.subtask-A.english.INPUT.txt', 'r')
lines = f.readlines()
total=len(lines)
train=0.8*total
count=0
for line in lines:
    if(count<train):
        x='train'
    else:
        x='test'
    count=count+1
    #print(count)
    temp = line.split("\t")
    meta = temp[0] + '\t' + x + '\t' + temp[1]
    meta_data_list.append(meta)
    tweets.append(normalizeTweet(temp[2]))
f.close()



meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(tweets)

f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()

f = open('data/corpus/' + dataset_name + '.clean.txt', 'w')
f.write(corpus_str)
f.close()