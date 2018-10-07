import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys

#from BilSTRM_CRF import BiLSTM_CRF
from helper import *

from preprocessor import Preprocessor
from BiLSTM import BiLSTM

torch.manual_seed(1)


EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

train = Preprocessor("data/train.txt","data/word_embeddings.txt")

word_to_ix = {}
for sentence, tags in train.batch:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B-TAR": 0, "B-HYP": 1,"I-TAR": 2,"I-HYP": 3, "O": 4}

#Checking the result
#print("This is word_to_ix = \n", word_to_ix)
#print("\n\n\nThis is prepare_sequence = \n", prepare_sequence(sentence, word_to_ix))

# ------------------------------   Check Generate Batch   --------------------------------------
"""start = 0
BATCH_SIZE = 2

batch_data = train.generate_batch(start, BATCH_SIZE)
print(batch_data)
print("Separating x and y..")

sen_list = list()
tag_list = list()

for i, x in batch_data:
    sen_list.append(i)
    temp_tag = prepare_seq(x, tag_to_ix)
    tag_list.append(temp_tag)


new_data = torch.tensor(sen_list)
new_y = torch.tensor(tag_list)

print("\nCreating Model....\n")

model = BiLSTM(10, 1, batch_first = True) # Parameters = hidden_size, num_layers, dropout, batch_first

print("\nFeeding batch_data..\n")
print("\nThis is embedding dim truly = ", len(sen_list[0][0]))
print("\nThis is new_data = ", new_data)
print("\nThis is new_data.size = ", new_data.size())
result = model(new_data)
print("\n\nThis is the result = \n", result)"""


# ----------------------------   THE REAL TRAINING LOOP   --------------------------------------

lstm_model = BiLSTM(hidden_size = 10, num_layers = 1, batch_first = True)
loss_func = nn.NLLLoss()
print("This is the parameter = ", list(lstm_model.parameters()))
optimizer = optim.Adam(lstm_model.parameters(), lr = 0.0001, weight_decay = 0)

ITERATION = 100
BATCH_SIZE = 2

for nb in range(ITERATION):

    # Restart the batch in this loop

    start = 0

    for i in range(train.num_sentence()//BATCH_SIZE): # Calculate how many batches are there

        # Generate batch in this loop
        
        # Generate batch training data
        batch_train_data = train.generate_batch(start, BATCH_SIZE)

        # Separating x and y
        sen_list = list()
        tag_list = list()

        for i, x in batch_data:
            sen_list.append(i)
            temp_tag = prepare_seq(x, tag_to_ix)
            tag_list.append(temp_tag)

        new_data = torch.tensor(sen_list)
        new_y = torch.tensor(tag_list)

        # ----- TRAINING -----

        lstm_model.zero_grad() # PyTorch accumulates gradients, so we need to clear them out before each instances
        lstm_model.start_hidden() # Generate a random number for hidden_state

        # -Forward pass
        result = lstm_model(new_data)
        loss = loss_func(result, new_y) #Comparing result with new_y # result need to be adjusted
        loss.backward()
        optimizer.step()







#batch_data = list(list(list(embed)), list(tag))

"""
model = BiLSTM(hidden_size = HIDDEN_DIM, num_layers = 1, dropout = 0.5, batch_first = True)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(train.batch[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in train.batch[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in train.batch:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        #targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(train.batch[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!"""