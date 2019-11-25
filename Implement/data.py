import re
import gensim
import numpy as np
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

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

# dataset file
def get_data_from_dataset():
    train_data = []
    val_data = []
    test_data = []

    train_file = open('./liar_dataset/train.tsv','rb').read().decode('utf-8')
    val_file = open('./liar_dataset/valid.tsv','rb').read().decode('utf-8')
    test_file = open('./liar_dataset/test.tsv','rb').read().decode('utf-8')

    for line in train_file.split('\n'):
        train_data.append(line.strip().split('\t'))
        
    for line in val_file.split('\n'):
        val_data.append(line.strip().split('\t'))
    
    for line in test_file.split('\n'):
        test_data.append(line.strip().split('\t'))

    return train_data, val_data, test_data

def preprocess_statement(data):
    data = data.lower()
    new_data = re.sub('[^a-zA-Z0-9- ]', '', data)
    new_data = word_tokenize(new_data)

    # remove stopwords
    for _w in stopwords.words('english'):
        for _d in new_data:
            if _d == _w:
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
max_len = 47
def word2vec(str):
    vec = []

    for tok in preprocess_statement(str):
        if tok == " ":
            print("blank")
        temp_vec = []
        if tok not in voca.keys():
            temp_vec = np.array(voca['unknow'])
        else:
            temp_vec = np.array(voca[tok])
        vec.append(temp_vec)
    for l in range(max_len - len(vec)):
        vec.append(np.array(voca['unknow']))

    #reshape vec to 300*len
    # vec = np.reshape(vec, (300,max_len))

    return vec


# find max len for padding
# ignored_data = ['1606.json','1993.json','1720.json','1653.json','11191.json','40.json']
def find_max_len():
    max_len = 0

    for data in get_data_from_dataset():
        for sample in data:
            # if sample[0] not in ignored_data:
            _vec = word2vec(sample[2])
            _len = len(_vec)
            if _len > max_len:
                print(sample[0], "with len: ", _len)
                max_len = _len
            
    print("max len: ", max_len)

# preprocess meta-data
def data_to_dict(data, dict):
    for e in data:
            e = re.sub('[^a-zA-Z0-9- ]', '', e).lower()
            if e not in dict.keys():
                dict[e] = 1
            else:
                dict[e] += 1
    return dict

def freq_to_id(dict):
    dict_id = 0
    dict['other'] = len(dict) + 2
    for e in dict.keys():
        dict[e] = dict_id
        dict_id += 1
    return dict

def get_meta_dict():
    train_data, val_data, test_data = get_data_from_dataset()

    subject = {}
    speaker = {}
    job = {}
    state = {}
    party = {}
    context = {}

    for d in train_data + val_data + test_data:
        # if re.sub('[^a-zA-Z0-9- ]', '', d[6].strip().split('/')[0]).lower() == '':
        #     print(d[0])
        _subj = d[3].strip().split(',')
        _spe = d[4].strip().split(',')
        _job = d[5].strip().split(',')
        _st = d[6].strip().split('/')
        _pa = d[7].strip().split(',')
        if len(d) < 14:
            d.append('')
        _con = d[13].strip().split(',')
        
        subject = data_to_dict(_subj, subject)
        speaker = data_to_dict(_spe, speaker)
        job = data_to_dict(_job, job)
        state = data_to_dict(_st, state)
        party = data_to_dict(_pa, party)
        context = data_to_dict(_con, context)
        
    subject = freq_to_id(dict(sorted(subject.items(), key=lambda x: x[1], reverse=True)[:14]))
    speaker = freq_to_id(dict(sorted(speaker.items(), key=lambda x: x[1], reverse=True)[:17]))
    job = freq_to_id(dict(sorted(job.items(), key=lambda x: x[1], reverse=True)[1:13]))
    state = freq_to_id(dict(sorted(state.items(), key=lambda x: x[1], reverse=True)[1:17]))
    party = freq_to_id(dict(sorted(party.items(), key=lambda x: x[1], reverse=True)[:2] + sorted(party.items(), key=lambda x: x[1], reverse=True)[3:6]))
    context = freq_to_id(dict(sorted(context.items(), key=lambda x: x[1], reverse=True)[:13]))
    # print(context)

    return subject, speaker, job, state, party, context

def get_meta_id(data, dict):
    if data in dict.keys():
        return dict[data]
    else:
        return dict['other']

def preprocess_meta_data(data, subject_dict, speaker_dict, job_dict, state_dict, party_dict, context_dict):
    meta_vector = np.array([])
    meta_dict = {}

    subject_vec = [0] * len(subject_dict.keys())
    speaker_vec = [0] * len(speaker_dict.keys())
    job_vec = [0] * len(job_dict.keys())
    state_vec = [0] * len(state_dict.keys())
    party_vec = [0] * len(party_dict.keys())
    context_vec = [0] * len(context_dict.keys())
    history_vec = [int(i) for i in data[8:13]]



    if len(data) < 14:
        data.append('')

    for e in data[3].strip().split(','):
        subject_vec[get_meta_id(e, subject_dict)] = 1
    
    for e in data[4].strip().split(','):
        speaker_vec[get_meta_id(e, speaker_dict)] = 1

    for e in data[5].strip().split(','):
        job_vec[get_meta_id(e, job_dict)] = 1

    for e in data[6].strip().split('/'):
        state_vec[get_meta_id(e, state_dict)] = 1

    for e in data[7].strip().split(','):
        party_vec[get_meta_id(e, party_dict)] = 1

    for e in data[13].strip().split(','):
        context_vec[get_meta_id(e, context_dict)] = 1

    meta_vector = subject_vec + speaker_vec + job_vec + state_vec + party_vec + context_vec \
        #  + history_vec

    return [int(i) for i in meta_vector]


def data_to_batch(data, batch_size):
    data_return = []

    _data_count = int(len(data))
    batch = batch_size

    # get meta-data dict
    subject_dict, speaker_dict, job_dict, state_dict, party_dict, context_dict = get_meta_dict()

    for i in range(int(len(data)/batch_size) + 1):
        # check when final range of data
        if _data_count == 0:
            break
        if _data_count >= batch:
            _data_count = _data_count - batch
        else: 
            batch = _data_count

        meta = []
        statement = []
        label = {}
        label_2_class = []
        label_6_class = []
        for d in data[i*batch:(i+1)*batch]:
            meta.append(preprocess_meta_data(d, subject_dict, speaker_dict, job_dict, state_dict, party_dict, context_dict))
            # print('done yet')
            statement.append(word2vec(d[2]))
            label_2_class.append(get_output_dict_by_num_of_classes(2)[str(d[1]).lower()])
            label_6_class.append(get_output_dict_by_num_of_classes(6)[str(d[1]).lower()])

        label[2] = label_2_class
        label[6] = label_6_class

        data_return.append({'statement':statement, 'label':label, 'meta': meta})
    
    return data_return

def preprocess_data(batch_size):

    train_data, val_data, test_data = get_data_from_dataset()

    train_data = data_to_batch(train_data, batch_size)
    val_data = data_to_batch(val_data, batch_size)
    test_data = data_to_batch(test_data, batch_size)

    # print(test_data[0]['label'][6])
    # _train_data = open('./preprocessed_data/train_data.pickle','wb')
    # _val_data = open('./preprocessed_data/val_data.pickle','wb')
    # _test_data = open('./preprocessed_data/test_data.pickle','wb')

    # pickle.dump(train_data, _train_data)
    # pickle.dump(val_data, _val_data)
    # pickle.dump(test_data, _test_data)

    # _train_data.close()
    # _val_data.close()
    # _test_data.close()

    # print("\npreprocessed")

    return train_data, val_data, test_data



# create voca_vector from pre-trained word2vec
# voca_vec = {'unknow':np.zeros(300)}
# pre_trained_word2vec_path = "./GoogleNews-vectors-negative300.bin"
# pre_trained_word2vec = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_word2vec_path, binary=True)
# def create_voca_vec():
#     c = 0
#     for data in get_data_from_dataset():
#         for t in data:
#             print(c, " ")
#             c += 1
#             for w in preprocess_statement(t[2]):
#                 if w in pre_trained_word2vec.vocab:
#                     voca_vec[w] = pre_trained_word2vec.wv[w]
#                 else:
#                     print(t[0] + "\t" + w)
#     voca_vec_file = open("./voca/voca_vec.pickle","wb")
#     pickle.dump(voca_vec,voca_vec_file)
#     voca_vec_file.close()
# create_voca_vec()

# data = np.zeros(20)

# a = data[20:30]

# print(len(a))
