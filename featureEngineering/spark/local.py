# -*- coding: utf-8 -*-
import  configparser
import os
import sys
from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from collections import Counter
import datetime
from pyspark.sql.functions import udf
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import gc
import matplotlib.pyplot as plt

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

# pathFile="D:/douyinData/train_1w.txt"
pathFile="D:/douyinData/final_track2_train.txt"
rawRdd_train = sc.textFile(pathFile).map(lambda line : line.split('\t'))
print('finish read rdd, start to init action log rdd:')
actionLogRdd_train = rawRdd_train.map(
            lambda x :(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
                       int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))
sqlContext = SQLContext(sc)
labels=[('uid',typ.IntegerType()),
    ('user_city',typ.IntegerType()),
    ('item_id',typ.IntegerType()),
    ('author_id',typ.IntegerType()),
    ('item_city',typ.IntegerType()),
    ('channel',typ.IntegerType()),
    ('finish',typ.IntegerType()),
    ('like',typ.IntegerType()),
    ('music_id',typ.IntegerType()),
    ('device',typ.IntegerType()),
    ('time',typ.LongType()),
    ('duration_time',typ.IntegerType())]
actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])

df_train = sqlContext.createDataFrame(actionLogRdd_train, actionLogSchema)

print("------------1、通过时间戳获取年月日时分，(没有工作日特征，月日交叉表示节日特征,年份转化有问题)-----------------")

#作品发布时间-作品发布的最早时间,转化为day
time_min = df_train.select(fn.min(df_train['time'])).collect()
df_train=df_train.withColumn('time_day', ((df_train.time-fn.lit(time_min[0][0])) /fn.lit(3600 * 24)).cast(typ.IntegerType()))
'''
print('--------2、对duration_time 和time_day  进行处理  count、sum_like 、sum_finish')
df_new=df_train.select(["duration_time","time_day","finish","like"])
gdf1=df_new.groupBy("duration_time")
df2=gdf1.agg(fn.count("finish").alias("count"),fn.sum("finish").alias("sum_finish"),fn.sum("like").alias("sum_like"))
df2.show()
df2.toPandas().to_csv("D:/douyinData/duration_time_explore.csv",index=False)

gdf2=df_new.groupBy("time_day")
df3=gdf2.agg(fn.count("finish").alias("count"),fn.sum("finish").alias("sum_finish"),fn.sum("like").alias("sum_like"))
df3.show()
df3.toPandas().to_csv("D:/douyinData/time_day_explore.csv",index=False)
'''
print("对duration_time和time_day 根据finish、like进行分组")

def DurationLikeBin(x):
    if x <=2:
        return 1
    elif 2<x<=12:
        return 2
    elif 12<x<=15:
        return 3
    elif 15<x<=22:
        return 4
    elif 22<x<=42:
        return 5
    else:
        return 6
converDurationLikeBin=udf(lambda x :DurationLikeBin(x), typ.IntegerType())
df = df_train.withColumn("duration_time_bin_like", converDurationLikeBin(df_train.duration_time))
df.select("duration_time_bin_like").show(5)

def DurationFinishBin(x):
    if x <=2:
        return 1
    elif 2<x<=12:
        return 2
    elif 12<x<=26:
        return 3
    elif 26<x<=42:
        return 4
    else:
        return 5
converDurationFinishBin=udf(lambda x :DurationFinishBin(x), typ.IntegerType())
df = df_train.withColumn("duration_time_bin_finish", converDurationFinishBin(df_train.duration_time))
df.select("duration_time_bin_finish").show(5)

def TimeLikeBin(x):
    if x >=822:
        return 1
    elif 810<=x<822:
        return 2
    elif 781<=x<810:
        return 3
    elif 748<=x<781:
        return 4
    elif 726<=x<748:
        return 5
    elif 646<=x<726:
        return 6
    else:
        return 7

converTimeLikeBin=udf(lambda x :TimeLikeBin(x), typ.IntegerType())
df = df_train.withColumn("time_day_bin_like", converTimeLikeBin(df_train.time_day))
df.select("time_day_bin_like").show(5)


def TimeFinshBin(x):
    if x >=795:
        return 1
    elif 792<=x<795:
        return 2
    elif 632<=x<792:
        return 3
    else:
        return 4

converTimeFinshBinBin=udf(lambda x :TimeFinshBin(x), typ.IntegerType())
df = df_train.withColumn("time_day_bin_finish", converTimeFinshBinBin(df_train.time_day))
df.select("time_day_bin_finish").show(5)


# print("--------1、针对uid，authorid，musicid等组合的正负样本数量统计特征--------")
# print("交叉特征的正负样本数量统计")
# posneg_feats_list = []
# print('cross count')
# users = ['uid']
# authors = ['author_id', 'item_city', 'channel', 'music_id', 'device','time_day','item_pub_hour']
# posneg_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])
# posneg_feats_list.append(['uid','author_id', 'channel'])
# posneg_feats_list.append(['uid', 'author_id', 'music_id'])
# posneg_feats_list.append(['uid','author_id', 'channel','time_day'])
# posneg_feats_list.append(['uid', 'author_id', 'music_id','time_day'])
#
# print("计算以下交叉特征的正负样本比例")  #有2、3、4维的交叉特征
# print(posneg_feats_list)
#
# for i in range(len(posneg_feats_list)):
#     group_cols=posneg_feats_list[i]
#     new_feature = '_'.join(group_cols)
#     #计算df_train数据中正负样本的比例，test中直接拼接，为null则填充为0或者均值
#     #正负样本判定字段：like  finish
#     #d第一步，先拼接
#     print(new_feature)
#     if len(group_cols)==2:
#
#         print("开始处理2维交叉变量")
#         df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
#                                                          )
#
#     if len(group_cols)==3:
#
#         print("开始处理3维交叉变量")
#         df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
#                                                          ,df_train[group_cols[2]].cast(typ.StringType()))
#
#     if len(group_cols)==4:
#
#         print("开始处理4维交叉变量")
#         df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
#                                                          ,df_train[group_cols[2]].cast(typ.StringType()) ,df_train[group_cols[3]].cast(typ.StringType()))
#     for target in ["like","finish"] :
#         df3=df_train.groupby([new_feature]).count().withColumnRenamed('count',new_feature+'_count')
#         df4=df_train.where(df_train[target]==1).groupby([new_feature]).count().withColumnRenamed('count',new_feature+"_count_"+target+"_1")
#         df3=df3.join(df4,new_feature,'left').na.fill(0)
#         # print("两列相除:得到正样本的比例",target)
#         df3=df3.withColumn(new_feature+"_"+target+"_pos_neg",fn.col(new_feature+"_count_"+target+"_1")/fn.col(new_feature+'_count'))
#         df3=df3.drop(new_feature+"_count_"+target+"_1",new_feature+'_count')
#         # print("新的df_train",target)
#         df_train=df_train.join(df3,new_feature,"left")
#         # df_train.show()

