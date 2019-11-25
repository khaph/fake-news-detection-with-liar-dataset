import torch
import numpy as np
import argparse

from data import get_dict, create_data_sample, max_len
from train import train, check_acc
from data2 import preprocess_data

from model import CNN


# hyper papameters
kernel_size = [2,3,4]
statement_kernel_num = 128
batch_size = 64
embed_dim = 300
num_of_classes = 6
keep_prob = 0.6
epoch = 10
lr = 0.025
# for test only
model_name = 'cnn_with_statement_only'

# main function
def main():

    # preprocess data before train
    # create_data_sample(num_of_classes)

    output_size = num_of_classes

    model = CNN(kernel_size, statement_kernel_num, embed_dim, output_size, keep_prob)

    # get data to train
    train_data, val_data, test_data = preprocess_data(64)

    val_res= train(train_in, train_out, val_in, val_out, model, model_name, epoch, batch_size, lr)

    print("\nvalid result: ", val_res)

# test function
def test():

    # load model
    model = torch.load('./pretrained/' + model_name)
    model.eval()

    # get data to test
    train_in, train_out, val_in, val_out, test_in, test_out = get_data()

    val_acc, val_predict = check_acc(model, val_in, val_out)
    test_acc, test_predict = check_acc(model, test_in, test_out)
    
    print("\nvalid result: ", val_acc)
    print("\ntest result: ", test_acc)

    return test_predict.tolist(), test_out.tolist()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-action", "--action", help="train or test", choices= ['train','test'])
args = parser.parse_args()

switcher = {
    'train': main,
    'test': test,
}

switcher[args.action]()
