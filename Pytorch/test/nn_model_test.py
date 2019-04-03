import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import data_preprocess

rootDir = 'E:/conf_test/dnn_ctr/data'

result_dict = data_preprocess.read_criteo_data(rootDir + '/tiny_train_input.csv', rootDir + '/category_emb.csv')
#test_dict = data_preprocess.read_criteo_data(rootDir + '/tiny_test_input.csv', rootDir + '/category_emb.csv')

# print(result_dict['index'])
# print('**********************')
print(result_dict['value'][0])

index_list = result_dict['index']
print(len(index_list[0]))
