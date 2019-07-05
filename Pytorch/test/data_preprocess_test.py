
import sys
import math
import argparse
import hashlib, csv, math, os, pickle, subprocess

from utils.data_preprocess import  *

cate_dict = load_criteo_category_index("C:/conf_test/deepctr/data/category_emb.csv")
print(cate_dict)

#'./data/tiny_train_input.csv', './data/category_emb.csv'
def read_criteo_data(file_path, emb_file):
    result = {'label':[], 'index':[],'value':[],'feature_sizes':[]}
    cate_dict = load_criteo_category_index(emb_file) #[{}, {},...{}]
    for item in cate_dict:
        result['feature_sizes'].append(len(item))  #每个field的feature value 数量
    f = open(file_path,'r')
    for line in f:
        datas = line.strip().split(',')
        result['label'].append(int(datas[0]))
        indexs = [int(item) for item in datas[1:]]
        values = [1 for i in range(39)]
        result['index'].append(indexs)
        result['value'].append(values)

read_criteo_data('C:/conf_test/deepctr/data/tiny_train_input.csv',
                 'C:/conf_test/deepctr/data/category_emb.csv')