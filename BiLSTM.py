import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim


class BiLSTM(nn.Module):

	def __init__(self, hidden_size, num_layers, dropout = 0, batch_first = False):
		super(BiLSTM,self).__init__()
		#self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout 
		self.batch_first = batch_first

	
		
	def forward(self, input_data):
		#Parameters Explanation
		#num_layers = number of stacked LSTM cell
		#input_size = the size of expected input features (embedding dimension)
		#hidden_size = the size of features expected in hidden state (output class)
		max_length = len(input_data)
		emb_dim = list(input_data.size())[-1]

		rnn = nn.LSTM(input_size = emb_dim, hidden_size = self.hidden_size, num_layers = self.num_layers, dropout = self.dropout, bidirectional = True)

		#h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
		#c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
		ho = torch.randn(self.num_layers * 2, 3, self.hidden_size) 
		co = torch.randn(self.num_layers * 2, 3, self.hidden_size)

		print("\nThis is emb_dim = ", emb_dim)
		print("This is input_data.size() = ", input_data.size())
		print("This is input_data.view.size() = ", input_data.view(len(input_data[0]), max_length, -1).size())
		print("This is input_data.view.size(-1) = ", input_data.view(len(input_data[0]), max_length, -1).size(-1))
		print("This is (seqlen, batch_size, emb_dim) = ", (len(input_data[0]), 3, emb_dim))
		#input_data shape should be the same as (seq_len, batch, input_size)
		output, (hn,cn) = rnn(input_data.view(len(input_data[0]), max_length , -1), (ho, co))
		#this is the output dimension = (seq_len, batch_size, Hidden_dim * 2 (if bidirectional))

		#softmax
		softmax = nn.Softmax(dim = 2) #dim parameters means which dimension that softmax will be applied to 

		print("\nThis is the output = ", output)
		print("\n\nThis is the size of the output = ", output.size())
		print("\n\n")

		output2 = softmax(output)
		#this is the output2 dimension = same as output dimension

		print("\nThis is the output2 = ", output2)
		print("\nThis is the size of output2 = ", output2.size())

		#max from softmax
		outind = torch.argmax(output2, dim = 2) # dim parameters means which dimension that argmax will be applied to 
		#this is the outind dimension = (seq_len, batch_size) *NOTE = it's basically reduce the dim in the arguments

		print("\nThis is the outind = ", outind)
		print("\nThis is the size of outind = ", outind.size())

		return outind

		#process the output data according to needs (part of speech tagging)
		#return 


#Training the model
#model = BiLSTM()
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam()
"""
#Iteration Loop (Epoch)
epoch_num = 100
for epoch in range(epoch_num):

	#Batch training
	#for sentence, tags in training_data:

		#Step one clear gradient
		#model.zero_grad()

		#Step 2 get out input ready
		#sentence_in = prepare_sequence(sentence, word to idx)
		# targets = prepare_sequence(tags, tas to idx)

		#Step 3 forward pass
		#tag_scores = model(sentence_in)

		#Step 4 Loss function
		#loss = loss_function(tag_scores, targets)
		#loss.backward()	
		#optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
"""