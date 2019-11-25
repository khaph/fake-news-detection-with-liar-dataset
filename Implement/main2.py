from model import CNN
from hybrid_model import CNN_with_meta
import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from data2 import preprocess_data

kernel_size = [2,3,4]
statement_kernel_num = 128
batch_size = 1
statement_embed_dim = 300
meta_embed_dim = 83
keep_prob = 0.6
epoch = 5
lr = 0.0001
num_of_classes = 6
model_name = "model_"

# use meta or not
use_meta = True

if not use_meta: 
    model = CNN(kernel_size, statement_kernel_num, statement_embed_dim, num_of_classes, keep_prob)
    model_name += "statement_only"
else:
    model = CNN_with_meta(kernel_size, 
            statement_kernel_num, 
            statement_embed_dim,
            meta_embed_dim,
            num_of_classes, 
            keep_prob,
            meta_lstm_hidden_dim = 6,
            meta_lstm_num_of_layers = 2,
            meta_lstm_bidirectional = True)
    model_name += "statement_and_meta"

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    

optim = torch.optim.Adam(model.parameters(), lr)

def train(model, data):
    total_epoch_loss = 0
    total_epoch_acc = 0

    model.to('cpu')
    model.train()

    # print(len(train_data))

    for idx, batch in enumerate(data):
        data_in = batch
        target = torch.LongTensor(batch['label'][num_of_classes])
        # print(batch['meta'])

        optim.zero_grad()
        prediction = model(data_in)
        loss = F.cross_entropy(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()

        # print(num_corrects, len(batch['statement']))

        acc = 100.0 * num_corrects/len(batch['statement'])
        loss.backward()
        # clip_gradient(model, 1e-1)
        optim.step()
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(data), total_epoch_acc/len(data)

def eval_model(model, data):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data):
            data_in = batch
            target = torch.LongTensor(batch['label'][num_of_classes])

            prediction = model(data_in)
            loss = F.cross_entropy(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects/len(batch['statement'])
            # print(num_corrects, acc, len(batch['statement']), len(val_data))
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(data), total_epoch_acc/len(data)

print('\nPreprocessing...')
train_data, val_data, test_data = preprocess_data(batch_size)
print("Done!!")

print('\nTraining...')
for epoch in range(epoch):
    train_loss, train_acc = train(model, train_data)
    val_loss, val_acc = eval_model(model, val_data)
    
    print(f'\n[INFO]\tEpoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

test_loss, test_acc = eval_model(model, test_data)
print(f'\nTest Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

# save model
torch.save(model, "./pretrained/" + model_name + "_" + str(test_acc/100))
print("\nSaved model: " + model_name + "_" + str(test_acc/100))