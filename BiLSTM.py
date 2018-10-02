import numpy as np 
import torch.nn as nn


#Open the dataset


#Initialize BiLSTM Layer
bilstm = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1, dropout = 0, bidirectional = True)

#Feed data to BiLSTM
output, hidden = bilstm(data)

