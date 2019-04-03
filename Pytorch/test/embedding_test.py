import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)

# hello_idx = torch.LongTensor([word_to_ix['hello']])
# print(hello_idx)
# hello_idx = Variable(hello_idx)
# print(hello_idx)
# hello_embed = embeds(hello_idx)
# print(hello_embed)

# x = torch.randn(4, 5)
# print(x)
# print("***********")
# print(torch.sum(x, 0))#按列求和
# print("***********")
#
# y = torch.sum(x, 1)#按行求和
# print(y)

emb = nn.Embedding(10,1)

print(emb(Variable(torch.LongTensor([2]))))
print(emb(torch.LongTensor([2])))