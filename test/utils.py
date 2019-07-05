import  numpy as np
import  pandas as pd


def save_fieldIndex_dict(field_index_dict, file_path):
    f = open(file_path,'w')
    for i, index_dict in enumerate(field_index_dict):
        for key in index_dict:
            f.write(str(i)+','+key+','+str(index_dict[key])+'\n')
    f.close()