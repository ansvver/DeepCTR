import os
import  sys
import pandas as pd
import  math

from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
from pyspark.sql.types import *
import  configparser
import  pyspark.sql.types as types
from pyspark.sql import  functions
import  json

from pyspark.sql.functions import regexp_extract,col # regexp_extract是pyspark正则表达式模块
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer, OneHotEncoder, StringIndexer, IndexToString, VectorIndexer, VectorAssembler
from pyspark.ml.feature import ChiSqSelector, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression,LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

from spark_feature_engine import  SparkFeatureEngine

current_path = os.path.dirname(os.path.realpath(__file__))
workspace_path = current_path.split('featureEngineering')[0]
sys.path.append(os.path.join(workspace_path, 'test'))

"""
explore task任务：
1. 主要针对action log 等原始交互行为特征属性进行初步分析， 
   观察异常值，离群值等等，为连续变量 低稀疏类别变量进行离散化，one-hot编码提供参考
2. 针对高稀疏变量得到的统计特征，进一步分析，为连续变量进行离散化提供参考
"""


columns = [('uid', types.StringType()),
                  ('user_city', types.StringType()),
                  ('item_id', types.StringType()),
                  ('author_id', types.StringType()),
                  ('item_city', types.StringType()),
                  ('channel', types.StringType()),
                  ('finish', types.StringType()),
                  ('like', types.StringType()),
                  ('music_id', types.StringType()),
                  ('device', types.StringType()),
                  ('time', types.StringType()),
                  ('duration_time', types.IntegerType())]

columns_withIndex = [('uid', types.StringType()),
                  ('user_city', types.StringType()),
                  ('item_id', types.StringType()),
                  ('author_id', types.StringType()),
                  ('item_city', types.StringType()),
                  ('channel', types.StringType()),
                  ('finish', types.StringType()),
                  ('like', types.StringType()),
                  ('music_id', types.StringType()),
                  ('device', types.StringType()),
                  ('time', types.StringType()),
                  ('duration_time', types.IntegerType()),
                  ('index', types.IntegerType())]

feature_dir = '/user/hadoop/icmechallenge2019/track2/test/features/original_statistic/'

'''
    duration time 特征的异常值分析
'''
def explore_durationTime(train_df):

    df = train_df.select(['uid', 'duration_time', 'finish', 'like'])

    # # 1.计算平均值
    # means_value = df.select(functions.mean('duration_time')).collect()[0][0]
    # # 2.计算方差
    # devs_value = df.select(((df.duration_time - means_value) ** 2).alias('deviation'))
    # # 3.计算标准差
    # stddev_value = math.floor(math.sqrt(devs_value.groupBy().avg('deviation').first()[0]))
    #
    # # 4.如果在区间内，则值正常，返回当前值；如果在区间外，则认定为异常值，返回均值
    # no_outlier = df.select(
    #     df.duration_time,
    #     functions.when(df.duration_time.between(means_value - 2 * stddev_value, means_value + 2 * stddev_value),
    #                    df.duration_time)
    #         .otherwise(means_value)
    #         .alias("updated_salary"),
    #     df.like,
    #     df.finish
    # )
    # no_outlier.show()
    #
    # no_outlier.toPandas().to_csv('durationTime_explore_data.csv', index=None)

    # 5. groupby duration time
    cnt_df = df.groupby('duration_time').agg({'uid' : 'count'})
    sum_df = df.groupby('duration_time').agg({'finish' :'sum', 'like':'sum'})

    cnt_df = cnt_df.withColumnRenamed('count(uid)', 'cnt_durationTime')
    sum_df = sum_df.withColumnRenamed('sum(finish)', 'sum_finish')
    sum_df = sum_df.withColumnRenamed('sum(like)', 'sum_like')

    join_df = cnt_df.join(sum_df, on = 'duration_time', how = 'inner')

    join_df.toPandas().to_csv('explore_data/durationTime_explore_data.csv', index = None)

'''
观察交互行为数据中music_id的原始统计特征
'''
def explore_music_statistic_feature():
    print('start to load music dataframe:')
    music_feature_rdd = spark_job.read_rdd(feature_dir + 'music')

    columns = [('music_id', types.StringType()),
               ('music_cnt', types.FloatType()),
               ('music_sum_like', types.FloatType()),
               ('music_sum_finish', types.FloatType()),
               ('music_finish_rate', types.FloatType()),
               ('music_like_rate', types.FloatType())]

    music_feature_rdd = music_feature_rdd.map(lambda line: line.split('\t')) \
        .map(lambda x: (x[0], float(x[1]), float(x[2]),
                        float(x[3]), float(x[4]), float(x[5])))

    schema = types.StructType([types.StructField(e[0], e[1], True) for e in columns])

    music_feature_df = spark_job.rdd2df(rdd=music_feature_rdd, schema=schema)

    rate_df = music_feature_df.select('music_id', 'music_finish_rate', 'music_like_rate')

    finishRate_statistic_df = rate_df.groupby('music_finish_rate').agg({'music_id': 'count'})
    finishRate_statistic_df = finishRate_statistic_df.withColumnRenamed('count(music_id)', 'finish_rate_cnt')

    finishRate_statistic_df.show()

    likeRate_statistic_df = rate_df.groupby('music_like_rate').agg({'music_id': 'count'})
    likeRate_statistic_df = likeRate_statistic_df.withColumnRenamed('count(music_id)', 'like_rate_cnt')

    likeRate_statistic_df.show()

    return (finishRate_statistic_df.toPandas(), likeRate_statistic_df.toPandas())


'''
    观察交互行为数据中device_id的原始统计特征
'''
def explore_device_statistic_feature():
    print('start to load device dataframe:')


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


if __name__ == '__main__':
    file = sys.argv[1]
    spark_engine = SparkFeatureEngine()
    train_actionLog_df = load_train_actionLog_df(file, columns)

    explore_durationTime(train_actionLog_df)