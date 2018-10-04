import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from BilSTRM_CRF import BiLSTM_CRF
from helper import *

from preprocessor import Preprocessor

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
# embed_data = train.embed_input("data/word_embeddings.txt")
train.generate_batch(0,2)

word_to_ix = {}
for sentence, tags in train.batch:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B-TAR": 0, "B-HYP": 1,"I-TAR": 2,"I-HYP": 3, "O": 4}

#Checking the result
#print("This is train result = \n", train.batch)
#print("\n\n\nThis is word to ix = \n", word_to_ix)

"""
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
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
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

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