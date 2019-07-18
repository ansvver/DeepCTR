import os
import  sys
from  pyspark.sql import  *
from pyspark.sql.types import *
import  pyspark.sql.types as types
from pyspark.sql import  functions
from pyspark.sql.functions import udf, col, isnull
from pyspark.ml.feature import StandardScaler
import  json
import  math
import  numpy as np


sys.path.append("..")
current_path = os.path.dirname(os.path.realpath(__file__))
print(current_path)

from spark_feature_engine import  SparkFeatureEngine
import  config
from config import  *



'''
actionLog
basic feature rdd:
['uid', 'item_id', 'author_id', 'finish', 'like', 'music_id', 
'device',  'index', 'duration_time_bins', 'user_city_encode', 
'item_city_encode', 'channel_encode']
'''

'''
actionLog
statistic feature rdd:
['uid', 'user_finish_ratio', 'user_like_ratio', 'user_finish_ratio_bins', 'user_like_ratio_bins', 
            'distinct_cnt_authorId_bins', 'distinct_cnt_musicId_bins']
['music_id', 'music_finish_ratio', 'music_like_ratio', 'music_finish_ratio_bins', 'music_like_ratio_bins']
['item_id', 'item_finish_ratio', 'item_like_ratio', 'item_finish_ratio_bins', 'item_like_ratio_bins']
['device', 'device_finish_ratio', 'device_like_ratio', 'device_finish_ratio_bins', 'device_like_ratio_bins']
['author_id', 'author_finish_ratio', 'author_like_ratio', 'author_finish_ratio_bins', 'author_like_ratio_bins']
'''
def load_feature_df(type, file, columns):
    schema = types.StructType([types.StructField(e[0], e[1], True) for e in columns])
    raw_rdd = spark_engine.read_rdd(file).map(lambda line: line.split('\t'))
    if type == 'music':
        rdd = raw_rdd.map(lambda x: (x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4])))
    elif type == 'author':
        rdd = raw_rdd.map(lambda x: (x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4])))
    elif type == 'device':
        rdd = raw_rdd.map(lambda x: (x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4])))
    elif type == 'item':
        rdd = raw_rdd.map(lambda x: (x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4])))
    elif type == 'user':
        rdd = raw_rdd.map(lambda x: (x[0], float(x[1]), float(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6])))
    elif type == 'basic':
        rdd = raw_rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], int(x[5]), int(x[6]),
                                     int(x[7]), int(x[8]), int(x[9]), x[10], x[11]))

    return  spark_engine.rdd2df(rdd, schema)

def deepfm_data_generate():
    print('start to generate train data for deepfm model ---> finish job')

    print('load original statistic data:')
    music_statistic_df = load_feature_df('music', config.hdfs_feature_root_dir + 'statistic/music',
                                         config.music_schema)
    author_statistic_df = load_feature_df('author', config.hdfs_feature_root_dir + 'statistic/author',
                                          config.author_schema)
    item_statistic_df = load_feature_df('item', config.hdfs_feature_root_dir + 'statistic/item',
                                        config.item_schema)
    device_statistic_df = load_feature_df('device', config.hdfs_feature_root_dir + 'statistic/device',
                                          config.device_schema)
    user_statistic_df = load_feature_df('user', config.hdfs_feature_root_dir + 'statistic/user',
                                        config.user_schema)

    print('load basic feature data:')
    basic_feature_df = load_feature_df('basic', config.hdfs_feature_root_dir + 'basic/category_continues_features',
                                       config.basic_feature_schema)
    '''
    join feature datas
    '''
    train_data_df = basic_feature_df.join(music_statistic_df, on = 'music_id', how = 'left')
    train_data_df = train_data_df.join(user_statistic_df, on='uid', how='left')
    train_data_df = train_data_df.join(author_statistic_df, on = 'author_id', how = 'left')
    train_data_df = train_data_df.join(item_statistic_df, on = 'item_id', how = 'left')
    train_data_df = train_data_df.join(device_statistic_df, on = 'device', how = 'left')

    '''
    sava train data
    '''
    deepfm_features = ['finish'] + [c for c in train_data_df.columns if c not in ['uid','index', 'music_id','author_id', 'device',
                                                                   'item_id', 'uid','like','finish','music_finish_ratio',
                                                                    'music_like_ratio','user_finish_ratio','user_like_ratio',
                                                                   'author_finish_ratio','author_like_ratio','item_finish_ratio',
                                                                   'item_like_ratio','device_finish_ratio','device_like_ratio']]

    deepfm_train_data = train_data_df.select(deepfm_features)


    # for col in train_data_df.columns:
    #     null_df = train_data_df.filter(isnull(col))
    #     print('col: ' + col + ', null count:')
    #     print (null_df.rdd.count())


    hdfs_train_data_dir = config.hdfs_feature_root_dir + 'train_data/deepfm/finish'
    local_train_dfata_dir = config.local_feature_root_dir + 'train_data/deepfm'

    deepfm_train_data.show()
    print('train_data_df: ' + str(deepfm_train_data.count()))
    print('train data fields:')
    print(deepfm_train_data.columns)


    os.system("hadoop fs -rmr {}".format(hdfs_train_data_dir))
    col_num = len(deepfm_train_data.columns)
    print (col_num)
    train_data_rdd = deepfm_train_data.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    train_data_rdd.coalesce(1, True).saveAsTextFile(hdfs_train_data_dir)
    os.system("hadoop fs -get {} {}".format(hdfs_train_data_dir, local_train_dfata_dir))
    print ('merge deepfm train data finished!')

def read_train_data(file_path, field_size):
    result = {'label': [], 'index': [], 'value': [], 'feature_sizes':[]}
    cate_dict = []
    for i in range(field_size):
        cate_dict.append(set())
    f = open(file_path, 'r')
    for line in f:
        datas = line.strip().split('\t')
        result['label'].append(int(datas[0]))
        indexs = [int(item) for item in datas[1:]]
        values = [1 for i in range(field_size)]

        for i in range(1, field_size + 1):
            j = i - 1
            cate_dict[j].add(int(datas[i]))

        result['index'].append(indexs)
        result['value'].append(values)

    for item in cate_dict:
        max_index = max(item)
        result['feature_sizes'].append(max_index + 1)  #每个field的feature value 数量

    return  result


if __name__ == '__main__':
    spark_engine = SparkFeatureEngine()
    deepfm_data_generate()

