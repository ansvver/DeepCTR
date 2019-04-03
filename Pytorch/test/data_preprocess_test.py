
import sys
import math
import argparse
import hashlib, csv, math, os, pickle, subprocess

from utils.data_preprocess import  *

cate_dict = load_criteo_category_index("E:/conf_test/dnn_ctr/data/category_emb.csv")
print(cate_dict)