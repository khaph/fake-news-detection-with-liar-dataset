import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_with_meta(nn.Module):
    def __init__(self, 
        kernel_size, 
        statement_kernel_num, 
        statement_embed_dim,
        meta_embed_dim,
        output_size, 
        keep_prob,
        meta_lstm_hidden_dim = 5,
		meta_lstm_num_of_layers = 2,
		meta_lstm_bidirectional = True,
        ):

        super(CNN_with_meta, self).__init__()
        
        self.statement_kernel_num = statement_kernel_num
        self.kernel_size = kernel_size
        self.statement_embed_dim = statement_embed_dim
        self.meta_embed_dim = meta_embed_dim
        self.output_size = output_size
        self.meta_lstm_hidden_dim = meta_lstm_hidden_dim
        self.meta_lstm_num_of_layers = meta_lstm_num_of_layers
        self.meta_lstm_bidirectional = meta_lstm_bidirectional
        self.meta_lstm_num_of_directions = 2 if self.meta_lstm_bidirectional else 1

        self.meta_query_dim = self.meta_lstm_num_of_layers * self.meta_lstm_num_of_directions
        self.drop_out = nn.Dropout(keep_prob)

        self.label = nn.Linear(len(kernel_size) * self.statement_kernel_num + self.meta_query_dim, output_size)

        self.meta_LSTM = nn.LSTM(
			input_size = self.meta_embed_dim,
			hidden_size = self.meta_lstm_hidden_dim,
            num_layers = self.meta_lstm_num_of_layers,
			batch_first = True,
			bidirectional = self.meta_lstm_bidirectional
        )

        # self.meta_convs = nn.Conv2d(1, 16, (1, self.meta_embed_dim))

        # self.meta_dense = nn.Linear(95, output_size)

        self.statement_convs = nn.ModuleList()
        for _k in self.kernel_size:
            self.statement_convs.append(nn.Conv2d(1, self.statement_kernel_num, (_k, self.statement_embed_dim)))

    def max_convs(self, input, conv_layer):
        out = conv_layer(input)

        activ = F.relu(out.squeeze(3))

        max_pool = F.max_pool1d(activ,activ.size()[2]).squeeze(2)

        return max_pool

    def forward(self, data_in):
        statement = torch.Tensor(data_in['statement']).unsqueeze(1)
        meta = torch.Tensor(data_in['meta']).unsqueeze(1)
        # print(data_in['meta'])
        # meta_max_out = [self.max_convs(meta, self.meta_convs)]
        # meta_max_out = self.meta_dense(meta)
        # meta_max_out = [self.meta_LSTM(meta)]

        _, (meta_out,__) = self.meta_LSTM(meta)

        meta_max_out = F.max_pool1d(meta_out, self.meta_lstm_hidden_dim).view(1, -1)

        statement_max_out = [self.max_convs(statement, layer) for layer in self.statement_convs]
        statement_max_out = torch.cat(statement_max_out, 1)
        # print(len(statement_max_out))

        # all_features = meta_max_out + statement_max_out
        all_out = torch.cat((statement_max_out, meta_max_out), 1)

        fc_in = self.drop_out(all_out)

        out = self.label(all_out)
        
        return out

        