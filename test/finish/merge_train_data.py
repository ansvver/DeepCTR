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
from utils import save_fieldIndex_dict
import  config
from config import  *



'''
basic feature rdd:
['uid', 'item_id', 'author_id', 'finish', 'like', 'music_id', 
'device',  'index', 'duration_time_bins', 'user_city_encode', 
'item_city_encode', 'channel_encode']
'''

'''
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

def run():
    print('start to generate train data for finish model:')

    print('load original statistic data:')
    music_statistic_df = load_feature_df('music', config.feature_root_dir + 'original_statistic/music',
                                         config.music_schema)
    author_statistic_df = load_feature_df('author', config.feature_root_dir + 'original_statistic/author',
                                          config.author_schema)
    item_statistic_df = load_feature_df('item', config.feature_root_dir + 'original_statistic/item',
                                        config.item_schema)
    device_statistic_df = load_feature_df('device', config.feature_root_dir + 'original_statistic/device',
                                          config.device_schema)
    user_statistic_df = load_feature_df('user', config.feature_root_dir + 'original_statistic/user',
                                        config.user_schema)

    print('load basic feature data:')
    basic_feature_df = load_feature_df('basic', config.feature_root_dir + 'basic/category_continues_features',
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
    train_data_df = train_data_df.select([c for c in train_data_df.columns if c not in ['index', 'music_id',
                                                                                        'author_id', 'device',
                                                                                        'item_id', 'uid','like']])
    for col in train_data_df.columns:
        null_df = train_data_df.filter(isnull(col))
        print('col: ' + col)
        null_df.show(50)

    train_data_dir = config.feature_root_dir + 'train_data/finish'
    train_data_df.show()
    print('train_data_df: ' + str(train_data_df.count()))
    print('train data fields:')
    print(train_data_df.columns)
    os.system("hadoop fs -rmr {}".format(train_data_dir))
    col_num = len(train_data_df.columns)
    train_data_rdd = train_data_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    train_data_rdd.coalesce(1, True).saveAsTextFile(train_data_dir)

if __name__ == '__main__':
    spark_engine = SparkFeatureEngine()
    run()

