import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine,euclidean
from sentence_transformers import SentenceTransformer, util
import pickle
import scipy

#code for adding sbert edges starts from line number 518-577

if len(sys.argv) != 4:
	sys.exit("Use: python build_graph.py <dataset> <similarity> <threshold>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', '2017', '2018','wnut','md']
# build corpus
dataset = sys.argv[1]
similarity = sys.argv[2]
# threshold = sys.argv[3]
threshold = float(sys.argv[3])

print(similarity)
print(threshold)

if dataset not in datasets:
	sys.exit("wrong dataset name")

if((similarity!='cos') and (similarity!='euc')and (similarity != 'dot')):
    sys.exit("wrong similarity: please give cos or euc")

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

f = open('data/' + dataset + '.txt', 'r')
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
f = open('data/corpus/' + dataset + '.clean.txt', 'r', encoding="utf8")
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)


# train docs collection --- added ------
num_train = 0
doc_train_data = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r', encoding="utf8")
lines = f.readlines()
for line in lines:
    if num_train < len(doc_train_list):
        doc_train_data.append(line.strip())
        num_train = num_train + 1
f.close()



train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
# print(train_ids)
print("training data")
random.shuffle(train_ids)
# print(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
# print(test_ids)
print("testing data")
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
# f = open('data/' + dataset + '.test.index', 'w')
f = open("data/ind.{}.test.index".format(dataset), 'w')
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
f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w', encoding="utf8")
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

f = open('data/corpus/' + dataset + '_vocab.txt', 'w', encoding="utf8")
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w')
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
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
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
# print(y)

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
# print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
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

# word co-occurence with context windows
window_size = 20
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

row = []
col = []
weight = []

node_size = train_size + vocab_size + test_size

# pmi as weights
pmi_sc = []
num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    # pmi_sc.append(pmi)
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# np.savetxt('data/' + dataset + '.pmi_sc.csv', pmi_sc, delimiter=',')

# import csv
# with open('data/' + dataset + '.pmi_sc', 'w') as f_out:
#     writer = csv.writer(f_out)
#     writer.writerows(pmi_sc)


# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
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

tfidf_sc = []
for i in range(len(shuffle_doc_words_list)):
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
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        # tfidf_sc.append(freq * idf)
        doc_word_set.add(word)

# np.savetxt('data/' + dataset + '.tfidf.csv', tfidf_sc, delimiter=',')


# SBERT code start

from tqdm import tqdm
#First, the model is initialized. 'all-MiniLM-L6-v2' is chosen among the others because it has higher speed and accuracy for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')

#shuffle_doc_words_list contains all the tweets. Embeddings for all tweets are computed and stored in the tweet_embeddings list.
# tweet_embeddings = model.encode(shuffle_doc_words_list) # on all train + test data
# tweet_embeddings = model.encode(doc_train_data) # on training data only
tweet_embeddings = model.encode(doc_content_list) # on all data (train + test)

np.savetxt('data/' + dataset + '.sbert_emb.csv', tweet_embeddings, delimiter=',')
# np.savetxt('data/' + dataset + '.labels.csv', np.array(doc_labels), delimiter=',')
cos_sim_arr = []
euc_sim_arr = []
dot_prod_arr = []
new_edges = 0
if(similarity=='cos'):
    print("Cosine ", threshold)
    #cosine similarity is computed for all pairs of tweets and is stored in the cos-sim multi-dimensional list
    cos_sim = util.cos_sim(tweet_embeddings, tweet_embeddings)

    # for i in range(len(doc_content_list)):
    #     for j in range(i+1,len(doc_content_list)):
    #         cos_sim_arr.append(cos_sim[i][j].item())
    # np.savetxt('data/' + dataset + '.cos_sim.csv', cos_sim_arr, delimiter=',')

    sim_docs = []
    # var = np.arange(0,1000)
    # k=0
    lpa_mat_arr = []
    lpa_row = []
    lpa_col = []
    #iterate through every pair of tweets using the nested for loop and see which pair has similarity value above threshold
    for i in tqdm(range(len(doc_train_data))):
        for j in range(i+1,len(doc_train_data)):
            if(cos_sim[i][j] > threshold) and doc_labels[i] == doc_labels[j]:
            # if (cos_sim[i][j].item() > threshold) :
                # print(tweet_embeddings[i])
                new_edges = new_edges + 1
                if i < train_size:
                    row.append(i)
                    lpa_row.append(i)
                else:
                    row.append(i + vocab_size)
                    #The matrix entries are ordered as: Training tweets, Words, Test tweets.
                    #If it is training tweet, the index will be less than training set size and the index to the sparse adjacency matrix will be the same
                    #If it is test tweet, the index in the adjacency matrix will be offset by the total number of words
                if j < train_size:
                    col.append(j)
                    lpa_col.append(j)
                else:
                    col.append(j + vocab_size)
                weight.append(100 * cos_sim[i][j].item())
                # lpa_mat_arr.append(1)

                # s_l = shuffle_doc_name_list[i].split('\t')[2], shuffle_doc_name_list[j].split('\t')[2]
                # s_l = doc_labels[i], doc_labels[j]
                # sim_docs.append([i, shuffle_doc_words_list[i], shuffle_doc_words_list[j], s_l, cos_sim[i][j]])

                # if k < 1000 and i == var[k]:
                # # if k < 1000 :
                #     s_l= doc_labels[i],doc_labels[j]
                #     # sim_docs.append([k,doc_content_list[i],doc_content_list[j], s_l, cos_sim[i][j]])
                #     sim_docs.append([k,i,doc_train_data[i],j,doc_train_data[j], s_l, cos_sim[i][j].item()])
                #     k = k+1



elif(similarity=='euc'):
    print("Euclidean ", threshold)
    max = 0
    #For euclidean distance, first, the maximum value of euclidean distance is calculated among all pairs and stored in max
    for i in tqdm(range(len(doc_content_list))):
        for j in range(i+1,len(doc_content_list)):
            ei = tweet_embeddings[i]
            ej = tweet_embeddings[j]
            sim = euclidean(ei,ej)
            if(sim>max):
                max = sim

    #iterate through every pair of tweets using the nested for loop and see which pair has similarity value above threshold            
    for i in range(len(doc_content_list)):
        for j in range(i+1,len(doc_content_list)):
            ei = tweet_embeddings[i]
            ej = tweet_embeddings[j]
            euc_sim = euclidean(ei,ej)
            euc_sim_arr.append(euc_sim)
            #euclidean distance between each pair is calculated and is divided by the maximum euclidean distance, so as to bring it to the range 0-1
            norm = euc_sim/max
            # (1 - the normalized euclidean distance) is taken so that the similarity between similar tweets is high and others are low
            esim = 1-norm

            if(esim > threshold):
                new_edges = new_edges + 1
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size)
                if j < train_size:
                    col.append(j)
                else:
                    col.append(j + vocab_size)
                weight.append(sim)
    # np.savetxt('data/' + dataset + '.euc_sim.csv', euc_sim_arr, delimiter=',')
elif(similarity == 'dot'):
    print("Dot product")
    for i in tqdm(range(len(doc_content_list))):
        for j in range(i+1,len(doc_content_list)):
            dot_prod = np.dot(tweet_embeddings[i], tweet_embeddings[j])
            dot_prod_arr.append(dot_prod)
            new_edges = new_edges + 1
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)

            if j < train_size:
                col.append(j)
            else:
                col.append(j + vocab_size)
            weight.append(dot_prod)

    np.savetxt('data/' + dataset + '.dot_prod.csv', dot_prod_arr, delimiter=',')

# import csv
# with open(dataset + '.new_sim.csv', 'w') as f_out:
#     writer = csv.writer(f_out)
#     writer.writerows(sim_docs)

print("Number of new edges created: ", new_edges)
# weight = pmat.todense() + cmat.todense() + doc_mat.todense()
# adj = sp.csr_matrix((pmat.todense() + cmat.todense() + doc_mat.todense()))

# node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# lpa_adj = sp.csr_matrix(
#     (lpa_mat_arr, (lpa_row, lpa_col)), shape=(train_size, train_size))

graph_from_adj = nx.from_scipy_sparse_array(adj)
graph_dict_of_list = nx.to_dict_of_lists(graph_from_adj)
# print(adj)
print("Total number of edges",scipy.sparse.csr_matrix.getnnz(adj))

numerator = len(weight) + node_size
denominator = node_size * node_size
density = numerator / denominator
sparsity = 1 - density

print("Numerator: ", numerator," Denominator: ",denominator)
print("Density: ", density," Sparsity: ",sparsity)

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

# f = open("data/ind.{}.graph".format(dataset), 'wb')
# pkl.dump(lpa_adj, f)
# f.close()