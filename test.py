import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

ls_sentences = []


max_sentences = 2
count_sentence = 0

#read training data
fl_train = "data/train.txt"
fl = open(fl_train,"r")

ls_word = []
ls_tags = []
for line in fl:
    words = line.split()
    if (not words) :
        continue
    ls_word.append(words[0])
    ls_tags.append(words[1])

    if ('.' in line):
        # sentence ended here
        ls_sentences.append((ls_word,ls_tags))

        ls_word = []
        ls_tags = []
        count_sentence+=1
        if (count_sentence == max_sentences):
            count_sentence = 0
            break
    
    

# print(fl.read())

fl.close()
print(ls_sentences)