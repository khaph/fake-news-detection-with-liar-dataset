import pandas as pd
import gensim
import pickle
import torch
import re
import numpy as np

from nltk import tokenize as tk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from progressbar import display_bar

# dataset
train_data = pd.read_csv('liar_dataset/train.tsv', sep='\t', header=-1)
val_data = pd.read_csv('liar_dataset/valid.tsv', sep='\t', header=-1)
test_data = pd.read_csv('liar_dataset/test.tsv', sep='\t', header=-1)

# forbidden words
# forbidden_words_file = open("./forbidden_words.txt","r")
# forbidden_words = forbidden_words_file.read().split('\n')
# forbidden_words_file.close()

# stop words
_st = open("./stopwords_en.txt")
stop_words = _st.read().split("\n")
_st.close()

# get output dict
def get_output_dict_by_num_of_classes(_num_classes):
    output_labels_dict = {}
    if _num_classes == 6:
        output_labels_dict = {
            "pants-fire": 0,
            "false": 1,
            "barely-true": 2,
            "half-true": 3,
            "mostly-true": 4,
            "true": 5
        }
    else:
        output_labels_dict = {
            "pants-fire": 0,
            "false": 0,
            "barely-true": 0,
            "half-true": 1,
            "mostly-true": 1,
            "true": 1
        }
    return output_labels_dict

# remove stopwords, tokenize, remove forbidden words
# print(stop_words)

def preprocess(data):
    new_data = re.sub('[^a-zA-Z0-9- ]', '', data)
    new_data = word_tokenize(new_data)

    # remove stopwords
    for _w in stop_words:
        for _d in new_data:
            if _d.lower() == _w:
                new_data.remove(_d)
            
    return new_data

# load voca
def load_voca_from_pickle(path):
    pickle_file = open(path, "rb")
    voca = pickle.load(pickle_file)
    pickle_file.close()
    return voca

# word to vector
voca = load_voca_from_pickle('./voca/voca_vec.pickle')
max_len = 45
def word2vec(str):

    vec = []

    for tok in preprocess(str):
        temp_vec = []
        if tok not in voca.keys():
            temp_vec = np.array(voca['unknow'])
        else:
            temp_vec = np.array(voca[tok])
        vec.append(temp_vec)
    for l in range(max_len - len(vec)):
        vec.append(np.zeros(300))

    #reshape vec to 300*len
    # vec = np.reshape(vec, (300,max_len))

    return vec

# find max len for padding
ignored_data = ['1606.json','1993.json','1720.json','1653.json','11191.json','40.json']
def find_max_len():
    max_len = 0
    for sample in train_data.values + val_data.values + test_data.values:
        # if sample[0] not in ignored_data:
        _vec = word2vec(sample[2])
        _len = len(_vec)

        if _len > max_len:
            print(sample[0], "with len: ", _len)
            max_len = _len
            
    print("max len: ", max_len)
# find_max_len()


# create dictionary
train_in = []
train_out = []
test_in = []
test_out = []
val_in = []
val_out = []

def create_data_sample(num_of_classes):

    label_dict = get_output_dict_by_num_of_classes(num_of_classes)

    _sum = len(train_data.values) + len(val_data.values) + len(test_data.values)

    for t in train_data.values:
        train_in.append(word2vec(t[2]))
        _label = str(t[1]).lower()
        train_out.append(label_dict[_label])

        _cur = len(train_out) + len(val_out) + len(test_out)
        display_bar("Preprocessing: ", _cur, _sum)
        
    for t in val_data.values:
        val_in.append(word2vec(t[2]))
        _label = str(t[1]).lower()
        val_out.append(label_dict[_label])

        _cur = len(train_out) + len(val_out) + len(test_out)
        display_bar("Preprocessing: ", _cur, _sum)

    for t in test_data.values:
        test_in.append(word2vec(t[2]))
        _label = str(t[1]).lower()
        test_out.append(label_dict[_label])

        _cur = len(train_out) + len(val_out) + len(test_out)
        display_bar("Preprocessing: ", _cur, _sum)
    
    _train_in = open('./preprocessed_data/train_in.pickle','wb')
    _val_in = open('./preprocessed_data/val_in.pickle','wb')
    _test_in = open('./preprocessed_data/test_in.pickle','wb')
    _train_out = open('./preprocessed_data/train_out.pickle','wb')
    _val_out = open('./preprocessed_data/val_out.pickle','wb')
    _test_out = open('./preprocessed_data/test_out.pickle','wb')

    pickle.dump(train_in,_train_in)
    pickle.dump(val_in,_val_in)
    pickle.dump(test_in,_test_in)
    pickle.dump(train_out,_train_out)
    pickle.dump(val_out,_val_out)
    pickle.dump(test_out,_test_out)

    _train_in.close()
    _val_in.close()
    _test_in.close()
    _train_out.close()
    _val_out.close()
    _test_out.close()

# create_data_sample(6)

# get dictionary
def get_dict(name):

    dict_path = "./preprocessed_data/" + name + ".pickle"

    file = open(dict_path,'rb')
    dict = pickle.load(file)
    file.close()

    return dict

# get count of word
def get_count_of_word():
    return len(voca)


# create voca_vector from pre-trained word2vec
# voca_vec = {'unknow':np.zeros(300)}
# pre_trained_word2vec_path = "./GoogleNews-vectors-negative300.bin"
# pre_trained_word2vec = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_word2vec_path, binary=True)
# def create_voca_vec():
#     c = 0
#     for t in train_data.values + test_data.values + val_data.values:
#         print(c, " ")
#         c += 1
#         for w in preprocess(t[2]):
#             if w in pre_trained_word2vec.vocab:
#                 voca_vec[w] = pre_trained_word2vec.wv[w]
#     voca_vec_file = open("./voca/voca_vec.pickle","wb")
#     pickle.dump(voca_vec,voca_vec_file)
#     voca_vec_file.close()
# create_voca_vec()

# t1 = train_data.values
# t2 = train_data.values
# k = t1==t2
# print(k)

# print(t1)

print((train_data.values.tolist()+test_data.values.tolist()+val_data.values.tolist())[1])