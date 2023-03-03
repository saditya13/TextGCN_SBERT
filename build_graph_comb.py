import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm, preprocessing
from nltk.corpus import wordnet as wn
import nltk    #added
nltk.download('wordnet')
nltk.download('omw-1.4') #added
nltk.download('punkt') #added
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
from sklearn import preprocessing
import gensim
import gensim.downloader
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity # added by aditya
from nltk.tokenize import sent_tokenize, word_tokenize

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph_w2v.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'md','wnut','olid']
# build corpus
dataset = sys.argv[1]


if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset + '.txt', 'r', encoding = 'utf8')
# f = open('data/' + dataset + '.txt', 'r', encoding = 'utf8')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
# print(doc_train_list)
# print(doc_test_list)


# doc label storage
doc_labels = []
f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    doc_labels.append(temp[2])
f.close()

doc_content_list = []
clean_data = []  # chirag

f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
    words_in_line = line.strip().split(" ")  # chirag
    clean_data.append(words_in_line)  # chirag
f.close()


# print(doc_content_list)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)  #commented by aditya

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w',encoding='utf8')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
# print(ids)
print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + '_shuffle.txt', 'w', encoding='utf8')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w',encoding='utf8')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

# vocab_list = [[val] for val in vocab]

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w',encoding='utf8')
f.write(vocab_str)
f.close()


'''
word2vec pe-trained starts
'''
model_w2v = KeyedVectors.load_word2vec_format('G:/My Drive/Research works/twignet/pretrained word2vec models/GoogleNews-vectors-negative300.bin', binary = True)

vectors = []
# vector_word_map = []
for i, word in enumerate(vocab):
  try:
    vec = model_w2v[word]
  except KeyError:
    vec = np.array([0]*300)
  word_vector_map[word] = vec
word_embeddings_dim = model_w2v.vector_size
'''
word2vec pe-trained ends
'''
# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w', encoding='utf8')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w',encoding='utf8')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
print(ty)

# allx: the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

print(" length of vocab ", len(vocab))
# word co-occurence with context windows
window_size = 20
#window_size = sys.argv[2]
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
node_size = train_size + vocab_size + test_size
row1 = []
col1 = []
weight = []
print("computing pmi score")
# pmi as weights
pmi_sc= []
num_window = len(windows)
from tqdm import tqdm
for key in tqdm(word_pair_count):
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row1.append(train_size + i)
    col1.append(train_size + j)
    pmi_sc.append(pmi) # added

min_p = min(pmi_sc)
max_p = max(pmi_sc)
#normalization of pmi score between 0 and 1: added
print("nomralizing pmi")
for i, n in tqdm(enumerate(pmi_sc)):
    pmi_sc[i] = (n - min_p)/ (max_p - min_p)

pmat = sp.csr_matrix(
    (pmi_sc, (row1, col1)), shape=(node_size, node_size))
# print("pmi score matrix", pmat)


'''
# SBERT embeddings start
'''
from sentence_transformers import SentenceTransformer, util, models
from tqdm import tqdm
model_sbert = SentenceTransformer('../pretrained models/all-mpnet-base-v2')
print("computing cosine similarity between documents/tweets")
corpus_embeddings = model_sbert.encode(doc_content_list)
row4 = []
col4 = []
doc_sim = []
sim_labels = []
for i in tqdm(range(len(corpus_embeddings))):
  for j in range(len(corpus_embeddings)):
    veci = corpus_embeddings[i]
    vecj = corpus_embeddings[j]
    sim = util.cos_sim(veci, vecj).item()
    if sim > 0.5:
      s_l = doc_labels[i], doc_labels[j]
      sim_labels.append(s_l)
      row4.append(i)
      col4.append(j)
      doc_sim.append(sim)

doc_emb = sp.csr_matrix(
    (doc_sim, (row4, col4)), shape=(node_size, node_size))

total_count = 0
count_of_equals = 0

for i in range(len(sim_labels)):
    total_count = total_count+1
    if(sim_labels[i][0] == sim_labels[i][1]):
        count_of_equals = count_of_equals + 1

percent = (count_of_equals/total_count)*100
print("Label similarity percentage",percent)



'''
sbert embeddings end
'''




# word2vec similarity computation. added by aditya
print("computing cosine similarity")
row2= []
col2= []
cos_sim = []
similar_words = []
for i in tqdm(range(vocab_size)):
    for j in range(vocab_size):
        vector_i = np.array(word_vector_map[vocab[i]])
        vector_j = np.array(word_vector_map[vocab[j]])
        if not vector_i.any() or not vector_j.any():
          continue
        if i != j:
            sim = model_w2v.similarity(vocab[i],vocab[j])
            # sim = cosine_similarity([vector_i], [vector_j])[0][0]
            if sim > 0.5:
                row2.append(train_size + i)
                col2.append(train_size + j)
                cos_sim.append(sim)
                w = vocab[i], vocab[j]
                similar_words.append(w)


cmat = sp.csr_matrix(
    (cos_sim, (row2, col2)), shape=(node_size, node_size))
# print("cosine similarity matrix",cmat)
weight_dense = (0.5 * pmat.todense()) + (0.5 * cmat.todense())


# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

print("computing tf-idf score")
row3 = []
col3 = []
doc_weight = []
for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row3.append(i)
        else:
            row3.append(i + vocab_size)
        col3.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        doc_weight.append(freq * idf)
        doc_word_set.add(word)
#normalization between 0 and 1: added
min_d = min(doc_weight)
max_d = max(doc_weight)
print("normalizing tf-idf score")
for i, n in tqdm(enumerate(doc_weight)):
    doc_weight[i] = (n - min_d)/ (max_d - min_d)

doc_mat = sp.csr_matrix(
    (doc_weight, (row3, col3)), shape=(node_size, node_size))

adj = sp.csr_matrix(weight_dense + doc_mat.todense() + doc_emb.todense())

adj_nnz_count = adj.count_nonzero()
# print (adj.data)
sparsity = 1 - ( adj_nnz_count/ (node_size * node_size))
print("Current sparsity is",sparsity)


# dump objects
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

