import pandas as pd
import pickle
# from data import preprocess as pre
# from data import *
# from main import *

# key_not_in_word_embedding = open("./dataset_checking/data_not_in_word_embedding.txt","a")

# for sample in train_data.values.tolist() + val_data.values.tolist() + test_data.values.tolist():
#     list_of_tokens = []
#     for tok in preprocess(sample[2]):
#         if tok not in voca.keys():
#             list_of_tokens.append(tok)
#     if len(list_of_tokens) > 0:
#         key_not_in_word_embedding.write(sample[0])
#         print(sample[0])
#         for _t in list_of_tokens:
#             key_not_in_word_embedding.write("\t" + _t)
#             print("\t" + _t)
#         key_not_in_word_embedding.write("\n")
# key_not_in_word_embedding.close()



# test_output_6_classes = open("./dataset_checking/test_output_6_classes.txt","a")
# test_predict, test_out = test()

# id_to_text_output_dict = {
#     0: "pants-fire",
#     1: "false",
#     2: "barely-true",
#     3: "half-true",
#     4: "mostly-true",
#     5: "true"
# }
# count = 0
# for i,sample in enumerate(test_data.values.tolist()):
#     if test_out[i] != test_predict[i]:
#         _c = 0
#         test_output_6_classes.write(sample[0] + "\t" + id_to_text_output_dict[test_out[i]] + "\t" + id_to_text_output_dict[test_predict[i]])
#         for tok in pre(sample[2]):
#             if tok not in voca.keys():
#                 _c = 1
#                 print(tok)
#                 test_output_6_classes.write("\t" + tok)
#         test_output_6_classes.write("\n")
#         count += _c

# test_output_6_classes.close()

# print(count/980)

# fi = open("./preprocessed_data/train_in.pickle",'rb')
# a = pickle.load(fi)
# fi.close()
# print(a[0])

