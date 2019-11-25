import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score



def train(train_in, train_out, val_in, val_out, model, model_name, epoch, batch_size, lr):
    train_in = Variable(train_in)
    train_out = Variable(train_out)

    val_in = Variable(val_in)
    val_out = Variable(val_out)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    criterion = nn.CrossEntropyLoss()

    model.to('cpu')

    print("\nTraining...\n")

    for _e in range(epoch):

        model.train()

        train_in.to('cpu')
        train_out.to('cpu')
        val_in.to('cpu')
        val_out.to('cpu')

        tr_loss = 0

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        output_train = model(train_in)
        # outout_val = model(val_in)

        loss_train = criterion(output_train, train_out)
        # loss_val = criterion(outout_val, val_out)

        train_losses.append(loss_train)
        # val_losses.append(loss_val)

        loss_train.backward()

        optimizer.step()

        tr_loss += loss_train.item()

        print("Epoch: ", _e + 1, "\t", "loss: ", loss_train)

    torch.save(model, "./pretrained/" + model_name)
    print("\nsaved model: " + model_name)

    # check accuracy
    val_acc = check_acc(model, val_in, val_out)

    return val_acc

def eval(model, data_in, data_out):
    model.eval()

    total_loss = 0.0
    correct_num = 0
    
    for x, y in dl:        
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        loss = F.cross_entropy(scores, y)
        
        total_loss += loss.item()
        y_pred = torch.max(scores, 1)[1]
        correct_num += (y_pred == y).sum().item()
    
    avg_loss = total_loss / len(dl.dataset)
    avg_acc = correct_num / len(dl.dataset)

    return avg_loss, avg_acc

def check_acc(model, data_in, data_out):
    with torch.no_grad():
        output = model(data_in)
    
    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    acc = accuracy_score(data_out, predictions)

    return acc, predictions

