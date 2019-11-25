import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN(nn.Module):
    def __init__(self, kernel_size, statement_kernel_num, embed_dim, output_size, keep_prob):
        super(CNN, self).__init__()
        
        self.statement_kernel_num = statement_kernel_num
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.output_size = output_size
        self.drop_out = nn.Dropout(keep_prob)
        self.label = nn.Linear(len(kernel_size)*self.statement_kernel_num, output_size)

        self.statement_convs = nn.ModuleList()
        for _k in self.kernel_size:
            self.statement_convs.append(nn.Conv2d(1,self.statement_kernel_num,(_k, self.embed_dim)))

    def maxp_statement_convs(self, input, conv_layer):
        out = conv_layer(input)

        activ = F.relu(out.squeeze(3))

        max_pool = F.max_pool1d(activ,activ.size()[2]).squeeze(2)

        return max_pool

    def forward(self, data_in):
        data = torch.Tensor(data_in['statement']).unsqueeze(1)

        max_statement_out = [self.maxp_statement_convs(data, layer) for layer in self.statement_convs]

        all_out = torch.cat((max_statement_out), 1)

        fc_in = self.drop_out(all_out)

        out = self.label(fc_in)
        
        return out

        