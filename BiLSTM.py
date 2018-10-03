import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim


class BiLSTM(nn.Module):

	def __init__(self, hidden_size, num_layers, dropout = 0, batch_first = False):
		super(BiLSTM,self).init()
		#self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout 
		self.batch_first = batch_first

	def embed_input(input_data, embedd_path):
		
		#change input_data into list of embedd
		#this is the path supposed to be = "data/word_embeddings.txt"
		with open(embedd_path, "r", encoding = "utf-8") as file:
			text = dict()

			for i in file:
				temp = i.split()
				text[temp[0]] = list(map(float,temp[1:]))

		#input_data in forms of = [[list of words, list of tags]. [list of words, list of tags], ... ], type list(list(list(), list()))
		for i in input_data:
			
			length = len(i[0])

			for idx in range(length):

				if i[idx] in text:
					i[idx] = text[i[idx]]
				else:
					i[idx] = text["<UNK_WORD>"]

		return input_data
		
	def forward(input_data):
		#Parameters Explanation
		#num_layers = number of stacked LSTM cell
		#input_size = the size of expected input features (number of words in sentences)
		#hidden_size = the size of features expected in hidden state (output class)
		rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, dropout = self.dropout, bidirectional = True)

		#h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
		#c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
		ho = torch.randn(self.num_layers * 2, input_data, self.hidden_size) 
		co = torch.randn(self.num_layers * 2, input_data, self.hidden_size)

		#input_data shape should be the same as (seq_len, batch, input_size)
		output, (hn,cn) = rnn(self.emb_input, (ho, co))

		#softmax
		softmax = nn.Softmax()
		output2 = softmax(output)

		#max from softmax
		outind = torch.argmax(output2, dim = 2)

		return outind

		#process the output data according to needs (part of speech tagging)
		#return 


#Training the model
#model = BiLSTM()
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam()

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
