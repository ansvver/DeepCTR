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

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

pathFile="C:/data/douyinData/train_1w.txt"

rawRdd_train = sc.textFile(pathFile).map(lambda line : line.split('\t'))
print('finish read rdd, start to init action log rdd:')
# actionLogRdd_train = rawRdd_train.map(
#     lambda x :(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
#                int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))
sqlContext = SQLContext(sc)
# labels=[('uid',typ.IntegerType()),
#     ('user_city',typ.IntegerType()),
#     ('item_id',typ.IntegerType()),
#     ('author_id',typ.IntegerType()),
#     ('item_city',typ.IntegerType()),
#     ('channel',typ.IntegerType()),
#     ('finish',typ.IntegerType()),
#     ('like',typ.IntegerType()),
#     ('music_id',typ.IntegerType()),
#     ('device',typ.IntegerType()),
#     ('time',typ.LongType()),
#     ('duration_time',typ.IntegerType())]
# actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
#
# df_train = sqlContext.createDataFrame(actionLogRdd_train, actionLogSchema)

print("duration_time应该根据喜欢和不喜欢来分箱")
print("查看duration_time的分布")

# duration_times_like = rawRdd_train.filter(lambda x: int(x[7])==1).map(
#     lambda x : int(x[11])).collect()
# duration_times_unlike = rawRdd_train.filter(lambda x: int(x[7])==0).map(
#     lambda x : int(x[11])).collect()
duration_times_like=rawRdd_train.map(
      lambda x :( int(x[7]), int(x[11])))
labels=[('like',typ.IntegerType()),
    ('duration_time',typ.IntegerType())]

Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
df = sqlContext.createDataFrame(duration_times_like, Schema)

df_count=df.groupBy(['duration_time','like']).count()
df_count.show()
#根据df_count画图，count作为列，duration_time作为行，like作为两个对比的列






#
# import matplotlib.pyplot as plt
# fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
# #第二个参数是柱子宽一些还是窄一些，越大越窄越密
# ax0.hist(duration_times_like,20,normed=1,histtype='bar',facecolor='yellowgreen')  #,alpha=0.75
# ##pdf概率分布图，一万个数落在某个区间内的数有多少个
# ax0.set_title('duration_like')
# ax1.hist(duration_times_unlike,20,normed=1,histtype='bar',facecolor='pink')  #,alpha=0.75,cumulative=True,rwidth=0.8
# #cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
# ax1.set_title("duration_unlike")
# fig.subplots_adjust(hspace=0.4)
# plt.show()


# plt.hist(duration_times, bins=20, color='lightblue', normed=False)
# fig = plt.gcf()
# fig.set_size_inches(16, 10)
# plt.savefig("duration_time.png",transparent=True,format='png')
# plt.show()

#
# print("------------1、通过时间戳获取年月日时分，(没有工作日特征，月日交叉表示节日特征,年份转化有问题)-----------------")
#
# #作品发布时间-作品发布的最早时间,转化为day
# time_min = df_train.select(fn.min(df_train['time'])).collect()
# df_train=df_train.withColumn('time_day', ((df_train.time-fn.lit(time_min[0][0])) /fn.lit(3600 * 24)).cast(typ.IntegerType()))
# df_train=df_train.withColumn('time_strDate',fn.from_unixtime(df_train.time , "yyyy-MM-dd HH:mm:ss"))
# #将 unix 格式的时间戳转换为指定格式的日期,提取小时
# df_train=df_train.withColumn('item_pub_month',fn.from_unixtime(df_train.time , "M").cast(typ.IntegerType()))
# df_train=df_train.withColumn('item_pub_day',fn.from_unixtime(df_train.time , "d").cast(typ.IntegerType()))
# df_train=df_train.withColumn('item_pub_hour',fn.from_unixtime(df_train.time , "k").cast(typ.IntegerType()))
# df_train=df_train.withColumn('item_pub_minute',fn.from_unixtime(df_train.time , "m").cast(typ.IntegerType()))
# print("查看month,day,hour,minute的提取是否正确")
# df_train.show(truncate=False)
# df_train=df_train.drop('time_strDate')
# df_train=df_train.drop('time')
#
# print("--------1、针对uid，authorid，musicid等组合的正负样本数量统计特征--------")
# print("交叉特征的正负样本数量统计")
# posneg_feats_list = []
#
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

