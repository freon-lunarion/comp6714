import torch
import torch.nn as nn
import torch.optim as optim

#a = torch.randn(4,4,5)
#print("a = \n", a)
#b = torch.max(a, 2)
#print("\n\nb = \n", b)
#print("\n\nThe shape of b = ", b[0].size())
#print("\n\nThe shape of a = ", a.size())

test = {1: '1', 2: '2', 3:'3', 4:'4', 5:'5'}
if 3 in test:
	print("works1")
		
if '1' in test:
	print("works2")