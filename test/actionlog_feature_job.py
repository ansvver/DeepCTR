import os
import  sys
from  pyspark.sql import  *
from pyspark.sql.types import *
import  pyspark.sql.types as types
from pyspark.sql import  functions
from pyspark.sql.functions import udf, col
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
功能: 特征工程pipeline 流水线job，执行一系列task
1. 完成基于用户原始行为日志数据的一系列特征体系构建，包括基础特征，统计特征，共现特征等
   几大类特征体系数据构建
2. nlp,face,video等多模态特征的衍生
3. 各类特征转换功能
'''
'''
功能: 特征工程pipeline 流水线job，执行一系列task
1. 完成基于用户原始行为日志数据的一系列特征体系构建，包括基础特征，统计特征，共现特征等
   几大类特征体系数据构建
2. nlp,face,video等多模态特征的衍生，转换等任务函数实现
3. 特征离散化，one-hot编码转换等
'''


def get_field_index(x, field_index_map):
    return field_index_map[x]

def field_index_function(df, field_index_map, origin_col, new_col):
    field_index_udf = udf(lambda x: get_field_index(x, field_index_map), StringType())
    df = df.withColumn(new_col, field_index_udf(df[origin_col]))
    return  df

'''
等频离散化： 通过二分法寻找最佳的区间索引值，左边闭合，右边开
'''
def index_binarySearch(x, bins):
    start = 0
    end = len(bins) - 1
    while start <= end:
        if x <= bins[start]:
            return  start
        if x >= bins[end]:
            return end

        middle = (start + end) // 2
        if x >= bins[middle]:
            start = middle + 1
        elif x < bins[middle]:
            end = end - 1

def discretization_functions(df, discretization_bins, original_col, new_col):
    discretization_udf = udf(lambda x: index_binarySearch(x, discretization_bins), IntegerType())
    df = df.withColumn(new_col, discretization_udf(df[original_col]))
    return  df

'''
col_array: np.array
percents: list of percent
'''
def get_discretizationBins(col_array, percents):
    bins = []
    for percent in percents:
        bins.append(np.percentile(col_array, percent))
    return  bins

"""
基础特征体系：
1.低稀疏类别变量
   low_sparse_features = ['user_city', 'item_city', 'channel']
2.连续变量离散化
   continuous_features = ['time', 'duration_time']

目标变量：
finish， like
"""
def generate_basic_features(df, feature_dir):

    '''
    1. 连续变量离散化
        duration time 离散化
    '''
    duration_time_bins = [2, 10, 20, 30, 45, 70, 100, 300]
    df = discretization_functions(df, duration_time_bins, 'duration_time', 'duration_time_bins')
    '''
    2. 类别变量进行one-hot编码转换，原始值对应到该field下的feature index值，并保存category_index_map文件
    '''
    #user city index
    user_city_index_map = {}
    user_city_distinct = df.select('user_city').rdd.map(lambda x : str(x[0])).distinct().collect()
    for item in user_city_distinct:
        if item not in user_city_index_map.keys():
            user_city_index_map[item] = len(user_city_index_map)
    df = field_index_function(df, user_city_index_map, 'user_city', 'user_city_encode')

    #item city index
    item_city_index_map = {}
    item_city_distinct = df.select('item_city').rdd.map(lambda x : str(x[0])).distinct().collect()
    for item in item_city_distinct:
        if item not in item_city_index_map:
            item_city_index_map[item] = len(item_city_index_map)
    df = field_index_function(df, item_city_index_map, 'item_city', 'item_city_encode')

    #channel index
    channel_index_map = {}
    channel_index_distinct = df.select('channel').rdd.map(lambda x : str(x[0])).distinct().collect()
    for item in channel_index_distinct:
        if item not in channel_index_map:
            channel_index_map[item] = len(channel_index_map)
    df = field_index_function(df, channel_index_map, 'channel', 'channel_encode')

    '''
    3. 存储特征结果文件
    '''
    basic_feature_list = []
    for item in config.basic_feature_schema:
        basic_feature_list.append(item[0])
    print(basic_feature_list)
    basic_feature_df = df.select([x for x in basic_feature_list])
    basic_feature_df.show(10)
    print('basic feature fields:')
    print(basic_feature_df.columns)
    os.system("hadoop fs -rmr {}".format(feature_dir + 'category_continues_features'))
    col_num = len(basic_feature_df.columns)
    basic_feature_rdd= basic_feature_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    basic_feature_rdd.coalesce(1, True).saveAsTextFile(feature_dir + 'category_continues_features')

    return  basic_feature_df

"""
统计特征体系：
主要是针对高稀疏类别变量进行相关统计特征计算
"""
def generate_user_statistic_features(df, feature_dir):

    '''
    用户行为相关的连续特征离散化
    '''

    #1. uid分组item_id数目, uid分组author_id数目,uid分组music_id数目:全部distinct count
    user_distinct_cnt_df = df.groupby('uid').agg(functions.countDistinct('item_id').alias('distinct_cnt_itemId'),
                          functions.countDistinct('author_id').alias('distinct_cnt_authorId'),
                          functions.countDistinct('music_id').alias('distinct_cnt_musicId'))

    user_distinct_cnt_df = user_distinct_cnt_df.select(['uid', 'distinct_cnt_authorId', 'distinct_cnt_musicId'])
    uid_authorIdCnt_bins = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    uid_musicIdCnt_bins = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    user_distinct_cnt_df = discretization_functions(user_distinct_cnt_df, uid_authorIdCnt_bins,
                                                    'distinct_cnt_authorId', 'distinct_cnt_authorId_bins')
    user_distinct_cnt_df = discretization_functions(user_distinct_cnt_df, uid_musicIdCnt_bins,
                                                    'distinct_cnt_musicId', 'distinct_cnt_musicId_bins')


    #2. uid 观看完播率，点赞率
    user_count_df = df.groupby('uid').agg({'finish':'count', 'like':'count'})
    user_count_df = user_count_df.withColumnRenamed('count(finish)', 'user_cnt_finish')
    user_count_df = user_count_df.withColumnRenamed('count(like)', 'user_cnt_like')
    user_sum_df = df.groupby('uid').agg({'finish': 'sum', 'like': 'sum'})
    user_sum_df = user_sum_df.withColumnRenamed('sum(finish)', 'user_sum_finish')
    user_sum_df = user_sum_df.withColumnRenamed('sum(like)', 'user_sum_like')

    user_action_df = user_count_df.join(user_sum_df, on = 'uid', how = 'inner')
    user_action_df = user_action_df.withColumn('user_finish_ratio', user_action_df.user_sum_finish * 1.0 / user_action_df.user_cnt_finish)
    user_action_df = user_action_df.withColumn('user_like_ratio', user_action_df.user_sum_like * 1.0 / user_action_df.user_cnt_like)



    print('user action features:')
    uid_finish_ratio_perecnts = [0,  50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(user_action_df.select('user_finish_ratio').rdd.collect()),
                                                 uid_finish_ratio_perecnts)
    user_action_df = discretization_functions(user_action_df, discretization_bins,
                                              'user_finish_ratio', 'user_finish_ratio_bins')
    print(discretization_bins)

    uid_like_ratio_perecnts = [0, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(user_action_df.select('user_like_ratio').rdd.collect()),
                                                 uid_like_ratio_perecnts)
    user_action_df = discretization_functions(user_action_df, discretization_bins,
                                               'user_like_ratio', 'user_like_ratio_bins')
    print(discretization_bins)

    user_action_df = user_action_df.select(['uid', 'user_finish_ratio', 'user_like_ratio', 'user_finish_ratio_bins', 'user_like_ratio_bins'])
    user_distinct_cnt_df = user_distinct_cnt_df.select(['uid', 'distinct_cnt_authorId_bins', 'distinct_cnt_musicId_bins'])

    user_action_df = user_action_df.join(user_distinct_cnt_df, on = 'uid', how = 'inner')

    user_action_df.show()
    print('user feature fields:')
    print(user_action_df.columns)
    os.system("hadoop fs -rmr {}".format(feature_dir + 'user'))
    col_num = len(user_action_df.columns)
    user_rdd = user_action_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    user_rdd.coalesce(1, True).saveAsTextFile(feature_dir + 'user')

def generate_item_statistic_features(df, feature_dir):
    '''
    item 相关的比例特征离散化
    '''
    #3. item_id分组uid数目，item_id被用户点击量，并且完整观看占比和点赞占比
    item_cnt_df = df.groupby('item_id').agg({'uid':'count'})
    item_cnt_df = item_cnt_df.withColumnRenamed('count(uid)', 'item_cnt')
    item_sum_df = df.groupby('item_id').agg({'finish':'sum', 'like':'sum'})
    item_sum_df = item_sum_df.withColumnRenamed('sum(finish)', 'item_sum_finish')
    item_sum_df = item_sum_df.withColumnRenamed('sum(like)', 'item_sum_like')

    item_df = item_cnt_df.join(item_sum_df, on = 'item_id', how = 'inner')
    item_df = item_df.withColumn('item_finish_ratio', item_df.item_sum_finish * 1.0 / item_df.item_cnt)
    item_df = item_df.withColumn('item_like_ratio', item_df.item_sum_like * 1.0 / item_df.item_cnt)


    print('item features:')
    item_finish_ratio_perecnts = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(item_df.select('item_finish_ratio').rdd.collect()),
                                                 item_finish_ratio_perecnts)
    print(discretization_bins)
    item_df = discretization_functions(item_df, discretization_bins,
                                       'item_finish_ratio', 'item_finish_ratio_bins')

    item_like_ratio_percents = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(item_df.select('item_like_ratio').rdd.collect()),
                                                 item_like_ratio_percents)
    print(discretization_bins)
    item_df = discretization_functions(item_df, discretization_bins,
                                       'item_like_ratio', 'item_like_ratio_bins')
    item_df = item_df.select(['item_id', 'item_finish_ratio', 'item_like_ratio', 'item_finish_ratio_bins', 'item_like_ratio_bins'])
    item_df.show()
    print('item feature fields:')
    print(item_df.columns)
    os.system("hadoop fs -rmr {}".format(feature_dir + 'item'))
    col_num = len(item_df.columns)
    item_rdd = item_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    item_rdd.coalesce(1, True).saveAsTextFile(feature_dir + 'item')

def generate_author_statistic_features(df, feature_dir):
    '''
    author 相关的比例特征离散化
    '''
    #4. author_id受欢迎程度：该作者被用户的点击量，并且完整观看占比和点赞占比
    author_cnt_df = df.groupby('author_id').agg({'uid':'count'})
    author_cnt_df = author_cnt_df.withColumnRenamed('count(uid)', 'author_cnt')
    author_sum_df = df.groupby('author_id').agg({'finish':'sum', 'like':'sum'})
    author_sum_df = author_sum_df.withColumnRenamed('sum(finish)', 'author_sum_finish')
    author_sum_df = author_sum_df.withColumnRenamed('sum(like)', 'author_sum_like')

    author_df = author_cnt_df.join(author_sum_df, on = 'author_id', how = 'inner')
    author_df = author_df.withColumn('author_finish_ratio', author_df.author_sum_finish * 1.0 / author_df.author_cnt)
    author_df = author_df.withColumn('author_like_ratio', author_df.author_sum_like * 1.0 / author_df.author_cnt)


    print('author features:')
    author_finish_ratio_perecnts = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(author_df.select('author_finish_ratio').rdd.collect()),
                                                 author_finish_ratio_perecnts)
    print(discretization_bins)
    author_df = discretization_functions(author_df, discretization_bins,
                                         'author_finish_ratio', 'author_finish_ratio_bins')

    author_like_ratio_percents = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(author_df.select('author_like_ratio').rdd.collect()),
                                                 author_like_ratio_percents)
    print(discretization_bins)
    author_df = discretization_functions(author_df, discretization_bins,
                                         'author_like_ratio', 'author_like_ratio_bins')
    author_df = author_df.select(['author_id', 'author_finish_ratio', 'author_like_ratio','author_finish_ratio_bins', 'author_like_ratio_bins'])
    author_df.show()
    print('author_df: ' + str(author_df.count()))
    print('author feature fields:')
    print(author_df.columns)
    os.system("hadoop fs -rmr {}".format(feature_dir + 'author'))
    col_num = len(author_df.columns)
    author_rdd = author_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    author_rdd.coalesce(1, True).saveAsTextFile(feature_dir + 'author')


'''
music 相关的比例特征离散化
'''
def generate_music_statistic_features(df, feature_dir):

    #5. music_id受欢迎程度: music被用户点击量，并且完整观看占比和点赞占比
    music_cnt_df = df.groupby('music_id').agg({'uid':'count'})
    music_cnt_df = music_cnt_df.withColumnRenamed('count(uid)', 'music_cnt')
    music_sum_df = df.groupby('music_id').agg({'finish':'sum', 'like':'sum'})
    music_sum_df = music_sum_df.withColumnRenamed('sum(finish)', 'music_sum_finish')
    music_sum_df = music_sum_df.withColumnRenamed('sum(like)', 'music_sum_like')

    music_df = music_cnt_df.join(music_sum_df, on = 'music_id', how = 'inner')
    music_df = music_df.withColumn('music_finish_ratio', music_df.music_sum_finish * 1.0 / music_df.music_cnt)
    music_df = music_df.withColumn('music_like_ratio', music_df.music_sum_like * 1.0 / music_df.music_cnt)


    print('music features:')
    music_finish_ratio_percents = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(music_df.select('music_finish_ratio').rdd.collect()),
                                                 music_finish_ratio_percents)
    print(discretization_bins)
    music_df = discretization_functions(music_df, discretization_bins,
                                        'music_finish_ratio', 'music_finish_ratio_bins')
    music_like_ratio_percents = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(music_df.select('music_like_ratio').rdd.collect()),
                                                 music_like_ratio_percents)
    print(discretization_bins)
    music_df = discretization_functions(music_df, discretization_bins,
                                         'music_like_ratio', 'music_like_ratio_bins')
    music_df = music_df.select(['music_id', 'music_finish_ratio', 'music_like_ratio', 'music_finish_ratio_bins', 'music_like_ratio_bins'])
    music_df.show()
    print('music_df: ' + str(music_df.count()))
    print('music feature fields:')
    print(music_df.columns)
    os.system("hadoop fs -rmr {}".format(feature_dir + 'music'))
    col_num = len(music_df.columns)
    music_rdd = music_df.rdd.map(lambda x : '\t'.join([str(x[i]) for i in range(col_num)]))
    music_rdd.coalesce(1, True).saveAsTextFile(feature_dir + 'music')

    '''
    device 相关的比例特征离散化
    '''

def generate_device_statistic_features(df, feature_dir):
    #device受欢迎程度:device上发布的item完整观看占比和点赞占比
    device_cnt_df = df.groupby('device').agg({'item_id' : 'count'})
    device_cnt_df = device_cnt_df.withColumnRenamed('count(item_id)', 'device_cnt')
    device_sum_df = df.groupby('device').agg({'finish':'sum', 'like':'sum'})
    device_sum_df = device_sum_df.withColumnRenamed('sum(finish)', 'device_sum_finish')
    device_sum_df = device_sum_df.withColumnRenamed('sum(like)', 'device_sum_like')

    device_df = device_cnt_df.join(device_sum_df, on = 'device', how = 'inner')
    device_df = device_df.withColumn('device_finish_ratio', device_df.device_sum_finish * 1.0 / device_df.device_cnt)
    device_df = device_df.withColumn('device_like_ratio', device_df.device_sum_like * 1.0 / device_df.device_cnt)


    print('device features:')
    device_finish_ratio_percents = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(device_df.select('device_finish_ratio').rdd.collect()),
                                                 device_finish_ratio_percents)
    print(discretization_bins)
    device_df = discretization_functions(device_df, discretization_bins,
                                        'device_finish_ratio', 'device_finish_ratio_bins')
    device_like_ratio_percents = [0, 25, 50, 75, 100]
    discretization_bins = get_discretizationBins(np.array(device_df.select('device_like_ratio').rdd.collect()),
                                                 device_like_ratio_percents)
    print(discretization_bins)
    device_df = discretization_functions(device_df, discretization_bins,
                                        'device_like_ratio', 'device_like_ratio_bins')

    device_df = device_df.select(['device','device_finish_ratio', 'device_like_ratio', 'device_finish_ratio_bins', 'device_like_ratio_bins'])

    device_df.show()
    print('device_df: ' + str(device_df.count()))
    print('device feature fields:')
    print(device_df.columns)
    os.system("hadoop fs -rmr {}".format(feature_dir + 'device'))
    col_num = len(device_df.columns)
    device_rdd = device_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    device_rdd.coalesce(1, True).saveAsTextFile(feature_dir + 'device')


"""
共现特征体系
   主要是uid,vid等之间的正负共现关系
"""



def build_train_input(train_actionLog_df):

    features_dict = {}
    features_dict['low_sparse'] = ['user_city', 'item_city', 'channel']
    features_dict['continuous'] = ['time', 'duration_time']
    high_sparse_features = ['uid', 'item_id', 'author_id', 'music_id', 'device']

    '''
    构建基础特征
    '''
    basic_feature_dir = config.feature_root_dir + 'basic/'
    basic_df = train_actionLog_df.select([c for c in ['user_city', 'item_city', 'channel', 'time', 'duration_time']])
    basic_feature_df = generate_basic_features(train_actionLog_df, basic_feature_dir)

    '''
    构建统计特征
    '''
    statistic_feature_dir = config.feature_root_dir + 'original_statistic/'
    sub_columns = high_sparse_features + ['like', 'finish']
    statistic_df = train_actionLog_df.select([c for c in sub_columns])
    generate_user_statistic_features(statistic_df, statistic_feature_dir)
    generate_author_statistic_features(statistic_df, statistic_feature_dir)
    generate_device_statistic_features(statistic_df, statistic_feature_dir)
    generate_item_statistic_features(statistic_df, statistic_feature_dir)
    generate_music_statistic_features(statistic_df, statistic_feature_dir)

    '''
    构建共现特征
    '''




def load_train_actionLog_df(file, columns):
    print('start to read data for rdd:')
    raw_rdd = spark_engine.read_rdd(file).map(lambda line: line.split('\t'))
    print('finish read rdd, start to init action log rdd:')
    train_actionLog_rdd = raw_rdd.map(
        lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
                    int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))

    total = train_actionLog_rdd.count()
    print('total count: ' + str(total))
    train_actionLog_Schema = types.StructType([types.StructField(e[0], e[1], True) for e in columns])
    train_actionLog_df = spark_engine.rdd2df(train_actionLog_rdd, schema=train_actionLog_Schema)

    return  train_actionLog_df

def load_train_actionLogWithIndex_df(file, columns):
    print('start to read data for rdd:')
    raw_rdd = spark_engine.read_rdd(file).map(lambda line: line.split('\t'))
    print('finish read rdd, start to init action log rdd:')
    train_actionLog_rdd = raw_rdd.map(
        lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
                    int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11]), int(x[12])))

    total = train_actionLog_rdd.count()
    print('total count: ' + str(total))
    train_actionLog_Schema = types.StructType([types.StructField(e[0], e[1], True) for e in columns])
    train_actionLog_df = spark_engine.rdd2df(train_actionLog_rdd, schema=train_actionLog_Schema)

    return  train_actionLog_df


if __name__ == '__main__':
    file = sys.argv[1]
    spark_engine = SparkFeatureEngine()
    train_actionLog_df = load_train_actionLogWithIndex_df(file, actionLog_schema_withIndex)

    #构建训练集特征数据
    build_train_input(train_actionLog_df)
